use std::{
    borrow::Cow,
    env,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

use super::pd_types::api_path;
use crate::{
    config::types::RetryConfig,
    core::{
        is_retryable_status, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry,
        WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::{GenerationRequest, InputIds, StringOrArray},
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils,
        streaming_utils::BreakerTrackedStream,
        RouterTrait,
    },
};

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
    prefill_admission: PrefillAdmission,
}

const MAX_PREFILL_ROOMS_ENV: &str = "SGLANG_PD_MAX_PREFILL_ROOMS_PER_WORKER";
const PREFILL_ROOM_QUEUE_SIZE_ENV: &str = "SGLANG_PD_PREFILL_ROOM_QUEUE_SIZE";
const PREFILL_ROOM_QUEUE_TIMEOUT_ENV: &str = "SGLANG_PD_PREFILL_ROOM_QUEUE_TIMEOUT_SECS";
const DEFAULT_PREFILL_ROOM_QUEUE_SIZE: usize = 32_768;
const DEFAULT_PREFILL_ROOM_QUEUE_TIMEOUT_SECS: u64 = 28_800;
const REQUEST_ID_HEADER: &str = "x-request-id";
const REQUEST_ID_BODY_KEY: &str = "rid";
const MAX_FORWARDED_REQUEST_ID_LEN: usize = 96;
const SSE_DONE_LINE_WITH_SPACE: &[u8] = b"data: [DONE]";
const SSE_DONE_LINE_WITHOUT_SPACE: &[u8] = b"data:[DONE]";

/// Incrementally recognizes the OpenAI SSE terminal event without treating a
/// `data: [DONE]` substring inside a JSON payload as the stream terminator.
///
/// Reqwest may split an SSE line across arbitrary HTTP body chunks, so the
/// detector retains only the current line. The buffer is bounded because a
/// line longer than either valid sentinel can never become a sentinel later.
#[derive(Debug, Default)]
struct SseDoneDetector {
    line: Vec<u8>,
    line_too_long: bool,
}

impl SseDoneDetector {
    fn observe(&mut self, chunk: &[u8]) -> bool {
        const MAX_CANDIDATE_LINE_LEN: usize = SSE_DONE_LINE_WITH_SPACE.len() + 1;

        for &byte in chunk {
            if byte == b'\n' {
                let line = self.line.strip_suffix(b"\r").unwrap_or(&self.line);
                let is_done = !self.line_too_long
                    && (line == SSE_DONE_LINE_WITH_SPACE || line == SSE_DONE_LINE_WITHOUT_SPACE);
                self.line.clear();
                self.line_too_long = false;
                if is_done {
                    return true;
                }
                continue;
            }

            if !self.line_too_long {
                if self.line.len() < MAX_CANDIDATE_LINE_LEN {
                    self.line.push(byte);
                } else {
                    self.line.clear();
                    self.line_too_long = true;
                }
            }
        }

        false
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefillAdmissionConfig {
    max_rooms_per_worker: usize,
    queue_size_per_worker: usize,
    queue_timeout: Duration,
}

impl PrefillAdmissionConfig {
    fn optional_env(name: &str) -> Result<Option<String>, String> {
        match env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(env::VarError::NotPresent) => Ok(None),
            Err(env::VarError::NotUnicode(_)) => Err(format!("{name} contains non-Unicode data")),
        }
    }

    fn parse_usize(name: &str, raw: &str) -> Result<usize, String> {
        raw.trim()
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer, got {raw:?}"))
    }

    fn from_values(
        max_rooms: Option<&str>,
        queue_size: Option<&str>,
        queue_timeout_secs: Option<&str>,
    ) -> Result<Option<Self>, String> {
        let Some(max_rooms) = max_rooms.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        let max_rooms_per_worker = Self::parse_usize(MAX_PREFILL_ROOMS_ENV, max_rooms)?;
        if max_rooms_per_worker == 0 {
            return Ok(None);
        }
        if max_rooms_per_worker > u32::MAX as usize {
            return Err(format!(
                "{MAX_PREFILL_ROOMS_ENV} must not exceed {}",
                u32::MAX
            ));
        }

        let queue_size_per_worker = match queue_size {
            Some(raw) if !raw.trim().is_empty() => {
                Self::parse_usize(PREFILL_ROOM_QUEUE_SIZE_ENV, raw)?
            }
            _ => DEFAULT_PREFILL_ROOM_QUEUE_SIZE,
        };
        let queue_timeout_secs = match queue_timeout_secs {
            Some(raw) if !raw.trim().is_empty() => raw.trim().parse::<u64>().map_err(|_| {
                format!("{PREFILL_ROOM_QUEUE_TIMEOUT_ENV} must be a positive integer, got {raw:?}")
            })?,
            _ => DEFAULT_PREFILL_ROOM_QUEUE_TIMEOUT_SECS,
        };
        if queue_timeout_secs == 0 {
            return Err(format!(
                "{PREFILL_ROOM_QUEUE_TIMEOUT_ENV} must be greater than zero"
            ));
        }

        Ok(Some(Self {
            max_rooms_per_worker,
            queue_size_per_worker,
            queue_timeout: Duration::from_secs(queue_timeout_secs),
        }))
    }

    fn from_env() -> Result<Option<Self>, String> {
        let max_rooms = Self::optional_env(MAX_PREFILL_ROOMS_ENV)?;
        let queue_size = Self::optional_env(PREFILL_ROOM_QUEUE_SIZE_ENV)?;
        let queue_timeout = Self::optional_env(PREFILL_ROOM_QUEUE_TIMEOUT_ENV)?;
        Self::from_values(
            max_rooms.as_deref(),
            queue_size.as_deref(),
            queue_timeout.as_deref(),
        )
    }
}

#[derive(Debug)]
struct PrefillWorkerAdmission {
    worker: Arc<str>,
    semaphore: Arc<Semaphore>,
    active_rooms: AtomicUsize,
    queued_requests: AtomicUsize,
}

impl PrefillWorkerAdmission {
    fn new(worker: &str, room_limit: usize) -> Self {
        Metrics::set_pd_prefill_admission_active_rooms(worker, 0);
        Metrics::set_pd_prefill_admission_queued_requests(worker, 0);
        Metrics::set_pd_prefill_admission_room_limit(worker, room_limit);
        Self {
            worker: Arc::from(worker),
            semaphore: Arc::new(Semaphore::new(room_limit)),
            active_rooms: AtomicUsize::new(0),
            queued_requests: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
struct PrefillRoomPermit {
    state: Arc<PrefillWorkerAdmission>,
    rooms: usize,
    _permit: OwnedSemaphorePermit,
}

impl PrefillRoomPermit {
    fn new(state: Arc<PrefillWorkerAdmission>, rooms: usize, permit: OwnedSemaphorePermit) -> Self {
        state.active_rooms.fetch_add(rooms, Ordering::AcqRel);
        Metrics::increment_pd_prefill_admission_active_rooms(state.worker.as_ref(), rooms);
        Self {
            state,
            rooms,
            _permit: permit,
        }
    }
}

impl Drop for PrefillRoomPermit {
    fn drop(&mut self) {
        let previous = self
            .state
            .active_rooms
            .fetch_sub(self.rooms, Ordering::AcqRel);
        debug_assert!(previous >= self.rooms);
        Metrics::decrement_pd_prefill_admission_active_rooms(
            self.state.worker.as_ref(),
            self.rooms,
        );
    }
}

#[derive(Debug)]
struct PrefillQueueGuard {
    state: Arc<PrefillWorkerAdmission>,
}

impl PrefillQueueGuard {
    fn try_new(state: Arc<PrefillWorkerAdmission>, limit: usize) -> Option<Self> {
        state
            .queued_requests
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                (queued < limit).then_some(queued + 1)
            })
            .ok()?;
        Metrics::increment_pd_prefill_admission_queued_requests(state.worker.as_ref());
        Some(Self { state })
    }
}

impl Drop for PrefillQueueGuard {
    fn drop(&mut self) {
        let previous = self.state.queued_requests.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        Metrics::decrement_pd_prefill_admission_queued_requests(self.state.worker.as_ref());
    }
}

#[derive(Debug)]
enum PrefillAdmissionError {
    RequestTooLarge {
        worker: Arc<str>,
        requested_rooms: usize,
        room_limit: usize,
    },
    QueueFull {
        worker: Arc<str>,
        queue_limit: usize,
    },
    Timeout {
        worker: Arc<str>,
        timeout: Duration,
    },
    Closed {
        worker: Arc<str>,
    },
}

#[derive(Debug)]
struct PrefillAdmission {
    config: Option<PrefillAdmissionConfig>,
    workers: DashMap<String, Arc<PrefillWorkerAdmission>>,
}

impl PrefillAdmission {
    fn new(config: Option<PrefillAdmissionConfig>) -> Self {
        Self {
            config,
            workers: DashMap::new(),
        }
    }

    fn from_env() -> Result<Self, String> {
        Ok(Self::new(PrefillAdmissionConfig::from_env()?))
    }

    fn state_for(
        &self,
        worker: &str,
        config: PrefillAdmissionConfig,
    ) -> Arc<PrefillWorkerAdmission> {
        self.workers
            .entry(worker.to_string())
            .or_insert_with(|| {
                Arc::new(PrefillWorkerAdmission::new(
                    worker,
                    config.max_rooms_per_worker,
                ))
            })
            .clone()
    }

    async fn acquire(
        &self,
        worker: &str,
        requested_rooms: usize,
    ) -> Result<Option<PrefillRoomPermit>, PrefillAdmissionError> {
        let Some(config) = self.config else {
            return Ok(None);
        };
        let requested_rooms = requested_rooms.max(1);
        let state = self.state_for(worker, config);

        if requested_rooms > config.max_rooms_per_worker {
            Metrics::record_pd_prefill_admission_decision(worker, "request_too_large");
            return Err(PrefillAdmissionError::RequestTooLarge {
                worker: Arc::clone(&state.worker),
                requested_rooms,
                room_limit: config.max_rooms_per_worker,
            });
        }
        let requested_rooms_u32 = requested_rooms as u32;

        match Arc::clone(&state.semaphore).try_acquire_many_owned(requested_rooms_u32) {
            Ok(permit) => {
                Metrics::record_pd_prefill_admission_decision(worker, "admitted_immediate");
                return Ok(Some(PrefillRoomPermit::new(state, requested_rooms, permit)));
            }
            Err(TryAcquireError::NoPermits) => {}
            Err(TryAcquireError::Closed) => {
                Metrics::record_pd_prefill_admission_decision(worker, "closed");
                return Err(PrefillAdmissionError::Closed {
                    worker: Arc::clone(&state.worker),
                });
            }
        }

        let Some(queue_guard) =
            PrefillQueueGuard::try_new(Arc::clone(&state), config.queue_size_per_worker)
        else {
            Metrics::record_pd_prefill_admission_decision(worker, "queue_full");
            return Err(PrefillAdmissionError::QueueFull {
                worker: Arc::clone(&state.worker),
                queue_limit: config.queue_size_per_worker,
            });
        };

        let wait_start = Instant::now();
        let acquire = Arc::clone(&state.semaphore).acquire_many_owned(requested_rooms_u32);
        let result = tokio::time::timeout(config.queue_timeout, acquire).await;
        let waited = wait_start.elapsed();
        drop(queue_guard);

        match result {
            Ok(Ok(permit)) => {
                Metrics::record_pd_prefill_admission_wait(worker, "admitted", waited);
                Metrics::record_pd_prefill_admission_decision(worker, "admitted_after_wait");
                Ok(Some(PrefillRoomPermit::new(state, requested_rooms, permit)))
            }
            Ok(Err(_)) => {
                Metrics::record_pd_prefill_admission_wait(worker, "closed", waited);
                Metrics::record_pd_prefill_admission_decision(worker, "closed");
                Err(PrefillAdmissionError::Closed {
                    worker: Arc::clone(&state.worker),
                })
            }
            Err(_) => {
                Metrics::record_pd_prefill_admission_wait(worker, "timeout", waited);
                Metrics::record_pd_prefill_admission_decision(worker, "timeout");
                Err(PrefillAdmissionError::Timeout {
                    worker: Arc::clone(&state.worker),
                    timeout: config.queue_timeout,
                })
            }
        }
    }
}

struct PreparedWorkerRequest<'a> {
    endpoint_url: String,
    body: Cow<'a, Value>,
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
}

/// Marker placed on a `Response` by paths inside
/// `execute_dual_dispatch_internal` that have already recorded prefill and
/// decode breaker outcomes against the workers' actual per-side results
/// (rather than the final response status). The outer dispatcher reads this
/// and skips its own status-based `record_outcome` calls so a decode-only
/// transport failure can't be misattributed to a healthy prefill.
#[derive(Clone, Copy)]
struct BreakerOutcomesRecorded;

impl PDRouter {
    fn worker_endpoint_url(worker: &dyn Worker, endpoint: &str) -> String {
        api_path(worker.base_url(), endpoint)
    }

    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();

        if let Some(worker) = workers.first() {
            self.proxy_to_worker(worker.as_ref(), endpoint, headers)
                .await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker: &dyn Worker,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = Self::worker_endpoint_url(worker, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        let prefill_admission = PrefillAdmission::from_env()?;
        if let Some(config) = prefill_admission.config {
            info!(
                max_rooms_per_worker = config.max_rooms_per_worker,
                queue_size_per_worker = config.queue_size_per_worker,
                queue_timeout_secs = config.queue_timeout.as_secs(),
                "PD Prefill-phase room admission enabled"
            );
        } else {
            info!("PD Prefill-phase room admission disabled");
        }
        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            enable_igw: ctx.router_config.enable_igw,
            prefill_admission,
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn handle_prefill_admission_error(error: PrefillAdmissionError) -> Response {
        match error {
            PrefillAdmissionError::RequestTooLarge {
                worker,
                requested_rooms,
                room_limit,
            } => {
                warn!(
                    worker = worker.as_ref(),
                    requested_rooms,
                    room_limit,
                    "PD request exceeds the per-Prefill-worker room limit"
                );
                error::create_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "prefill_room_request_too_large",
                    format!(
                        "Request needs {requested_rooms} Prefill rooms, but worker {worker} has a {room_limit}-room limit"
                    ),
                )
            }
            PrefillAdmissionError::QueueFull {
                worker,
                queue_limit,
            } => {
                warn!(
                    worker = worker.as_ref(),
                    queue_limit, "PD Prefill room queue is full"
                );
                error::create_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "prefill_room_queue_full",
                    format!(
                        "Router Prefill room queue for worker {worker} reached {queue_limit} requests"
                    ),
                )
            }
            PrefillAdmissionError::Timeout { worker, timeout } => {
                warn!(
                    worker = worker.as_ref(),
                    timeout_secs = timeout.as_secs(),
                    "Timed out waiting for a PD Prefill room"
                );
                // This is an intentional overload response, not an upstream
                // failure. 429 prevents the generic 5xx retry loop from
                // immediately re-enqueueing the same request.
                error::create_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "prefill_room_queue_timeout",
                    format!(
                        "Timed out after {:.3}s waiting for a Prefill room on worker {worker}",
                        timeout.as_secs_f64()
                    ),
                )
            }
            PrefillAdmissionError::Closed { worker } => {
                error!(
                    worker = worker.as_ref(),
                    "PD Prefill admission semaphore unexpectedly closed"
                );
                error::service_unavailable(
                    "prefill_room_admission_closed",
                    format!("Prefill room admission closed for worker {worker}"),
                )
            }
        }
    }

    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    // Static key strings to avoid per-request allocations
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";
    const DISAGG_PREFILL_DP_RANK_KEY: &'static str = "disagg_prefill_dp_rank";

    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(prefill_worker.bootstrap_host());
                ports.push(prefill_worker.bootstrap_port());
                rooms.push(super::pd_types::generate_room_id());
            }
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            );
        } else {
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::from(prefill_worker.bootstrap_host()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                match prefill_worker.bootstrap_port() {
                    Some(v) => Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    /// Restore the stable replay/request correlation ID after the public
    /// OpenAI request has passed through the Router's typed protocol model.
    /// `openai-protocol` 1.0.0 does not model SGLang's `rid` extension, so the
    /// ordinary deserialize/serialize path drops it. The replay mirrors that
    /// value into `x-request-id`; inject it back into the worker JSON once,
    /// before the same value is cloned for Prefill and Decode.
    fn inject_request_id_into_value(
        mut original: Value,
        headers: Option<&HeaderMap>,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let Some(raw_request_id) = headers
            .and_then(|headers| headers.get(REQUEST_ID_HEADER))
            .and_then(|value| value.to_str().ok())
        else {
            return Ok(original);
        };
        let request_id = raw_request_id.trim();
        if !Self::is_safe_request_id(request_id) {
            debug!(
                request_id_length = request_id.len(),
                "Ignoring invalid x-request-id for SGLang rid propagation"
            );
            return Ok(original);
        }

        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;
        let rid = match batch_size {
            Some(size) if size > 1 => Value::Array(
                (0..size)
                    .map(|index| Value::from(format!("{request_id}-{index}")))
                    .collect(),
            ),
            _ => Value::from(request_id),
        };
        obj.insert(REQUEST_ID_BODY_KEY.to_string(), rid);
        Ok(original)
    }

    fn is_safe_request_id(request_id: &str) -> bool {
        !request_id.is_empty()
            && request_id.len() <= MAX_FORWARDED_REQUEST_ID_LEN
            && request_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
            })
    }

    fn inject_prefill_dp_rank_for_decode<'a>(
        decode_request: Cow<'a, Value>,
        prefill_worker: &dyn Worker,
    ) -> Result<Cow<'a, Value>, String> {
        let Some(prefill_dp_rank) = prefill_worker.dp_rank() else {
            return Ok(decode_request);
        };

        let mut decode_request = decode_request.into_owned();
        let Some(obj) = decode_request.as_object_mut() else {
            return Err(
                "Failed to insert disagg_prefill_dp_rank because request body is not an object"
                    .to_string(),
            );
        };

        obj.insert(
            Self::DISAGG_PREFILL_DP_RANK_KEY.to_string(),
            Value::from(prefill_dp_rank as u64),
        );
        Ok(Cow::Owned(decode_request))
    }

    async fn prepare_worker_request<'a>(
        route: &'static str,
        worker: &dyn Worker,
        json_request: Cow<'a, Value>,
    ) -> Result<PreparedWorkerRequest<'a>, String> {
        let body = if worker.is_dp_aware() {
            Cow::Owned(
                worker
                    .prepare_request(json_request.into_owned())
                    .await
                    .map_err(|err| {
                        format!(
                            "Failed to prepare request for worker {}: {}",
                            worker.url(),
                            err
                        )
                    })?,
            )
        } else {
            json_request
        };

        Ok(PreparedWorkerRequest {
            endpoint_url: Self::worker_endpoint_url(worker, route),
            body,
        })
    }

    async fn prepare_pd_worker_requests<'a>(
        route: &'static str,
        json_request: &'a Value,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<(PreparedWorkerRequest<'a>, PreparedWorkerRequest<'a>), String> {
        let prefill_request =
            Self::prepare_worker_request(route, prefill, Cow::Borrowed(json_request)).await?;
        let decode_json_request =
            Self::inject_prefill_dp_rank_for_decode(Cow::Borrowed(json_request), prefill)?;
        let decode_request =
            Self::prepare_worker_request(route, decode, decode_json_request).await?;

        Ok((prefill_request, decode_request))
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode) = match self
                            .select_pd_pair(
                                context.request_text.as_deref(),
                                context.model_id,
                                context.headers.as_ref(),
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        // Reserve the selected Prefill in routing load before
                        // waiting for its room permit. Cache-aware falls back
                        // to minimum load for cold prefixes; if Router-queued
                        // requests were invisible here, a full admission set
                        // would leave all workers tied and deterministic tie
                        // breaking could pile every later cold request onto one
                        // Prefill. This guard represents Router-side assigned
                        // work only: no P/D HTTP request or GPU allocation has
                        // happened yet. It is released at the same Prefill
                        // phase boundary as before, or immediately on any
                        // admission/preparation failure.
                        let prefill_guard =
                            WorkerLoadGuard::new(prefill.clone(), context.headers.as_ref());

                        // Keep excess work in the Router. The paired Decode
                        // request must not be sent until the selected Prefill
                        // worker has room, otherwise Decode allocates target
                        // KV/state while merely waiting for Prefill output.
                        let requested_rooms = context.batch_size.unwrap_or(1);
                        let prefill_room_permit = match self
                            .prefill_admission
                            .acquire(prefill.url(), requested_rooms)
                            .await
                        {
                            Ok(permit) => permit,
                            Err(e) => return Self::handle_prefill_admission_error(e),
                        };

                        let mut json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        json_request = match Self::inject_request_id_into_value(
                            json_request,
                            context.headers.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let ctx_is_stream = context.is_stream;
                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                prefill_guard,
                                prefill_room_permit,
                                start_time,
                            )
                            .await;

                        let status = response.status();
                        let outcomes_already_recorded = response
                            .extensions()
                            .get::<BreakerOutcomesRecorded>()
                            .is_some();
                        if !outcomes_already_recorded {
                            let not_error = status.is_success() || status.is_client_error();
                            // Prefill is always non-streaming and fully read before
                            // we get here, so its outcome is final.
                            prefill.record_outcome(not_error);
                            // Decode for a streaming request is still mid-flight at
                            // this point; the `BreakerTrackedStream` wrapped around
                            // its byte stream records the outcome on drop. Skip the
                            // eager success record to avoid masking "200-then-broken"
                            // decode workers.
                            if !ctx_is_stream {
                                decode.record_outcome(not_error);
                            }
                        }

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        decode: Arc<dyn Worker>,
        decode_guard: WorkerLoadGuard,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => match serde_json::from_slice::<Value>(&error_body) {
                    Ok(error_json) => {
                        json!({ "message": error_json, "status": status.as_u16() })
                    }
                    Err(parse_err) => {
                        let body_text = String::from_utf8_lossy(&error_body).to_string();
                        let preview: String = body_text.chars().take(256).collect();
                        tracing::warn!(
                            "Failed to parse decode error body as JSON from {}: {} \
                             (status={}, body preview: {:?})",
                            decode.url(),
                            parse_err,
                            status.as_u16(),
                            preview
                        );
                        json!({ "message": body_text, "status": status.as_u16() })
                    }
                },
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(response_headers),
                decode,
                decode_guard,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {}", e);
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        prefill_guard: WorkerLoadGuard,
        prefill_room_permit: Option<PrefillRoomPermit>,
        _start_time: Instant,
    ) -> Response {
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let (prepared_prefill, prepared_decode) = match Self::prepare_pd_worker_requests(
            context.route,
            &json_request,
            prefill.as_ref(),
            decode.as_ref(),
        )
        .await
        {
            Ok(requests) => requests,
            Err(e) => {
                error!("Failed to prepare PD worker requests: {}", e);
                return error::internal_error("pd_request_preparation_failed", e);
            }
        };

        // Prefill load is a phase load, not an end-to-end stream load. Decode
        // remains active until its non-streaming body is read or its streaming
        // response body is dropped.
        let mut decode_guard = Some(WorkerLoadGuard::new(decode.clone(), headers));

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            &prepared_prefill.endpoint_url,
            &prepared_prefill.body,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            &prepared_decode.endpoint_url,
            &prepared_decode.body,
            headers,
            false,
        );

        // Run both in this handler task (not a detached tokio::spawn) so a client
        // disconnect cancels the pending decode request too, keeping the
        // upstream-cancel behavior from #19524.
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let prefill_fut = prefill_request.send();
        let decode_fut = decode_request.send();
        tokio::pin!(prefill_fut);
        tokio::pin!(decode_fut);

        // Poll both until prefill resolves; decode normally resolves later, but
        // may resolve first if it rejects the request outright.
        let prefill_result;
        let mut decode_early: Option<Result<reqwest::Response, reqwest::Error>> = None;
        loop {
            tokio::select! {
                biased;
                pr = &mut prefill_fut => {
                    prefill_result = pr;
                    break;
                }
                dr = &mut decode_fut, if decode_early.is_none() => {
                    decode_early = Some(dr);
                }
            }
        }

        // SGLang emits the Prefill HTTP response only after its forward and KV
        // handoff have completed. Release both the routing load and the room
        // permit at that phase boundary, before waiting for Decode generation.
        // This is the key distinction from the old end-to-end global limiter.
        drop(prefill_guard);
        drop(prefill_room_permit);

        // Decode can't generate without prefill's KV, so any prefill failure
        // (non-2xx / transport error) dooms the paired decode request, which would
        // otherwise block in WaitingForInput until the 300s disaggregation
        // timeout. Drop the decode future to close its connection; the decode
        // engine then detects the disconnect and aborts the request in ~4-8s.
        let prefill_failed = match &prefill_result {
            Ok(resp) => !resp.status().is_success(),
            Err(_) => true,
        };

        if prefill_failed {
            warn!(
                "Prefill failed, aborting paired decode request decode_url={} prefill_url={}",
                decode.url(),
                prefill.url()
            );

            // Tick prefill by its real status (4xx = client fault). Don't record
            // decode: it was cancelled due to a prefill fault, not its own, so a
            // prefill error storm can't trip healthy decode breakers.
            let prefill_ok = match &prefill_result {
                Ok(r) => r.status().is_client_error(),
                Err(_) => false,
            };
            prefill.record_outcome(prefill_ok);

            // Status-faithful error shaping (4xx forwarded, transport/5xx -> 502).
            let mut response = match self
                .process_prefill_response(prefill_result, prefill.url(), false)
                .await
            {
                Err(error_response) => error_response,
                Ok(_) => error::bad_gateway(
                    "prefill_server_error",
                    "Prefill reported failure but returned a success response".to_string(),
                ),
            };
            response.extensions_mut().insert(BreakerOutcomesRecorded);
            return response;
        }

        // Prefill ok: take decode's result, awaiting it if still pending.
        let decode_result = match decode_early {
            Some(dr) => dr,
            None => (&mut decode_fut).await,
        };

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!("Decode response status: {}", status);

                if !status.is_success() {
                    error!(
                        "Decode server returned error status decode_url={} status={}",
                        decode.url(),
                        status
                    );

                    // Per-worker breaker attribution before the synthetic 5xx
                    // response takes over. Prefill ran concurrently in the
                    // `tokio::join!`: tick it based on its actual response
                    // status, not on the decode-driven failure. For
                    // non-streaming the response carries no tracked stream
                    // so record decode's outcome here too — but treat 4xx
                    // as a client fault rather than a worker fault, matching
                    // the legacy outer-dispatcher rule and the streaming
                    // `BreakerTrackedStream` pre-mark in
                    // `create_streaming_response`. For streaming
                    // `handle_decode_error_response` wraps the synthetic
                    // error SSE in a `BreakerTrackedStream` that ticks
                    // decode on drop, so skip to avoid double-counting.
                    // Mark the response so the outer dispatcher skips its
                    // status-derived `record_outcome`.
                    let prefill_ok = match &prefill_result {
                        Ok(r) => {
                            let s = r.status();
                            s.is_success() || s.is_client_error()
                        }
                        Err(_) => false,
                    };
                    prefill.record_outcome(prefill_ok);
                    if !context.is_stream {
                        let decode_ok = status.is_success() || status.is_client_error();
                        decode.record_outcome(decode_ok);
                    }

                    let mut response = self
                        .handle_decode_error_response(
                            res,
                            &context,
                            decode,
                            decode_guard
                                .take()
                                .expect("Decode load guard must exist until response completion"),
                        )
                        .await;
                    response.extensions_mut().insert(BreakerOutcomesRecorded);
                    return response;
                }

                // Process prefill response
                let prefill_body = if context.return_logprob {
                    match self
                        .process_prefill_response(
                            prefill_result,
                            prefill.url(),
                            context.return_logprob,
                        )
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                } else {
                    // Even if we don't need logprobs, we should check prefill status
                    match self
                        .process_prefill_response(prefill_result, prefill.url(), false)
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                };

                if context.is_stream {
                    // Streaming response
                    let prefill_logprobs = if context.return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        context.return_logprob,
                        Some(response_headers),
                        decode,
                        decode_guard
                            .take()
                            .expect("Decode load guard must exist until response completion"),
                    )
                } else {
                    // Non-streaming response
                    if context.return_logprob {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            prefill_body,
                        )
                        .await
                    } else {
                        // Direct passthrough when no logprobs needed
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response = Response::new(Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                error::internal_error(
                                    "read_response_failed",
                                    "Failed to read response",
                                )
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "Decode request failed"
                );
                // Decode failed at TCP/transport level. No tracked
                // stream will ever wrap a response (streaming path) and
                // we shortcut past the outer non-streaming
                // `record_outcome` too — so record decode failure
                // directly. Prefill ran concurrently in the
                // `tokio::join!`: record its real per-worker outcome
                // (success on a 2xx/4xx send, failure on transport
                // error) so the decode-driven 502 doesn't penalise a
                // healthy prefill. Mark the response so the outer
                // dispatcher skips its status-derived `record_outcome`
                // and we don't double-count.
                decode.record_outcome(false);
                let prefill_ok = match &prefill_result {
                    Ok(res) => {
                        let s = res.status();
                        s.is_success() || s.is_client_error()
                    }
                    Err(_) => false,
                };
                prefill.record_outcome(prefill_ok);

                let mut response = error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
                response.extensions_mut().insert(BreakerOutcomesRecorded);
                response
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    /// Builds the text used for cache-aware routing of a chat request.
    ///
    /// This must reflect the *full* conversation (system prompt, prior turns,
    /// the current message and tool context) so that KV-cache prefix matching
    /// routes to the worker that actually shares the most prefix. Using only the
    /// first message ignores the conversation history that drives KV reuse in
    /// multi-turn chats. See https://github.com/sgl-project/sglang/issues/26263.
    ///
    /// Returns `None` when the conversation has no text to route on, preserving
    /// the prior behavior of not feeding an empty key into prefix matching.
    fn build_chat_request_text(body: &ChatCompletionRequest) -> Option<String> {
        // `extract_text_for_routing` walks every message (system, prior turns,
        // current message, tool content) and is the same routing text the regular
        // (non-PD) router uses, keeping cache-aware routing consistent across both.
        let text = body.extract_text_for_routing();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let prefill = Self::pick_worker_by_policy_arc(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            headers,
            hash_ring.clone(),
            "prefill",
        )
        .await?;

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
        )
        .await?;

        // Record worker selection metrics (Layer 3)
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );

        Ok((prefill, decode))
    }

    async fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        headers: Option<HeaderMap>,
        decode: Arc<dyn Worker>,
        decode_guard: WorkerLoadGuard,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Uses select! to race stream.next() against tx.closed() so that
        // when the client disconnects the upstream HTTP connection is dropped
        // promptly, allowing the engine to abort the request.
        // `biased;` drains a ready upstream chunk before observing client
        // disconnect, so a chunk already produced by reqwest reaches the
        // client (and the logprob merger) before we tear the loop down.
        //
        // The upstream stream is wrapped in `BreakerTrackedStream` so the
        // decode worker's circuit breaker is updated once on drop: success
        // on clean completion (`[DONE]` sentinel or `None`), failure on
        // stream error, neither on client disconnect. PD's pre-PR semantics
        // treated 4xx (client error) as not-a-worker-fault, so we only
        // pre-mark the wrapper as Errored on 5xx — `handle_decode_error_response`
        // synthesizes a single-chunk SSE error envelope that would otherwise
        // stream cleanly to None and record a spurious success.
        let mut tracked =
            BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_string());
        if !(status.is_success() || status.is_client_error()) {
            tracked.mark_errored();
        }
        let decode_for_log = decode.clone();
        tokio::spawn(async move {
            let mut done_detector = SseDoneDetector::default();
            loop {
                tokio::select! {
                    biased;
                    chunk_result = tracked.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                let is_done = done_detector.observe(&chunk);

                                let result = if return_logprob && prefill_logprobs.is_some() {
                                    Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                        .unwrap_or(chunk)
                                } else {
                                    chunk
                                };

                                // Mark the wrapper completed before the client
                                // send: upstream finished cleanly regardless of
                                // whether the client is still listening, and
                                // the worker deserves the success tick either
                                // way. `mark_completed` is a no-op once Errored
                                // is set, so the synthetic-error path is unaffected.
                                if is_done {
                                    tracked.mark_completed();
                                }

                                if tx.send(Ok(result)).is_err() {
                                    tracing::debug!(
                                        "Receiver dropped (likely client disconnect), \
                                        cancelling upstream PD stream"
                                    );
                                    break;
                                }

                                if is_done {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                // BreakerTrackedStream already logged the error
                                // and marked the terminal state as Errored so
                                // the worker's circuit breaker will tick on drop.
                                let _ = tx.send(Err(format!("Stream error: {}", e)));
                                break;
                            }
                            None => break,
                        }
                    }
                    _ = tx.closed() => {
                        tracing::info!(
                            "Client disconnected, cancelling upstream PD stream from {}",
                            decode_for_log.url()
                        );
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, decode_guard)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<bytes::Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };

        if !return_logprob {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        endpoint_url: &str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(endpoint_url).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = Self::worker_endpoint_url(prefill.as_ref(), "health_generate");
        let decode_url = Self::worker_endpoint_url(decode.as_ref(), "health_generate");
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client.get(&decode_url).send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            Self::build_chat_request_text(body)
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        error::bad_request(
            "pd_unsupported_embeddings",
            "PD mode does not support /v1/embeddings",
        )
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/classify; returning bad request");
        error::bad_request(
            "pd_unsupported_classify",
            "PD mode does not support /v1/classify",
        )
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};
    use crate::policies::{CacheAwareConfig, CacheAwarePolicy, SelectWorkerInfo};

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
            prefill_admission: PrefillAdmission::new(None),
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[test]
    fn test_sse_done_detector_requires_a_complete_data_line() {
        let mut detector = SseDoneDetector::default();
        let embedded = br#"data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"w.Write([]byte(\"data: [DONE]\\n\\n\"))"}}]}}]}

"#;

        assert!(!detector.observe(embedded));
        assert!(detector.observe(b"data: [DONE]\n\n"));
    }

    #[test]
    fn test_sse_done_detector_handles_chunk_boundaries_and_crlf() {
        let mut detector = SseDoneDetector::default();

        assert!(!detector.observe(b"data: [DO"));
        assert!(!detector.observe(b"NE]\r"));
        assert!(detector.observe(b"\n\r\n"));

        let mut no_space = SseDoneDetector::default();
        assert!(no_space.observe(b"data:[DONE]\n"));
    }

    #[test]
    fn test_sse_done_detector_rejects_prefixes_and_suffixes() {
        let mut detector = SseDoneDetector::default();

        assert!(!detector.observe(b"prefix data: [DONE]\n"));
        assert!(!detector.observe(b"data: [DONE] suffix\n"));
        assert!(detector.observe(b"data: [DONE]\n"));
    }

    #[test]
    fn test_chat_request_text_uses_full_conversation() {
        // Regression test for https://github.com/sgl-project/sglang/issues/26263
        // Cache-aware routing must build its text from the full conversation, not
        // just the first message, so that KV-cache prefix matching reflects what
        // the worker will actually process in a multi-turn chat.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "First question about apples."},
                {"role": "assistant", "content": "Apples are red."},
                {"role": "user", "content": "Follow up question about oranges."}
            ]
        }))
        .expect("valid chat request");

        let text = PDRouter::build_chat_request_text(&body)
            .expect("multi-message chat should produce routing text");

        assert!(
            text.contains("apples"),
            "routing text must include earlier turns, got: {text:?}"
        );
        assert!(
            text.contains("oranges"),
            "routing text must include later turns (not only the first message), got: {text:?}"
        );
    }

    #[test]
    fn test_chat_request_text_none_when_no_text() {
        // When the conversation carries no text content, no routing text should
        // be produced (None) rather than an empty string, preserving the prior
        // PD behavior. See https://github.com/sgl-project/sglang/issues/26263.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": ""}
            ]
        }))
        .expect("valid chat request");

        assert!(
            PDRouter::build_chat_request_text(&body).is_none(),
            "empty conversation text should produce None, not Some(\"\")"
        );
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_worker_endpoint_url_uses_base_url_for_dp_aware_worker() {
        let worker = DPAwareWorkerBuilder::new("http://prefill:30000", 2, 4)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();

        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "health_generate"),
            "http://prefill:30000/health_generate"
        );
        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "/v1/models"),
            "http://prefill:30000/v1/models"
        );
    }

    #[tokio::test]
    async fn test_prepare_pd_worker_requests_uses_dp_aware_rank() {
        let prefill = DPAwareWorkerBuilder::new("http://prefill:30000", 2, 4)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        let decode = DPAwareWorkerBuilder::new("http://decode:30001", 1, 4)
            .worker_type(WorkerType::Decode)
            .build();
        let request = json!({
            "prompt": "shared prefix",
            "max_tokens": 8,
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            "bootstrap_room": 1234,
        });

        let (prefill_request, decode_request) =
            PDRouter::prepare_pd_worker_requests("/v1/completions", &request, &prefill, &decode)
                .await
                .unwrap();

        assert_eq!(
            prefill_request.endpoint_url,
            "http://prefill:30000/v1/completions"
        );
        assert_eq!(prefill_request.body["data_parallel_rank"], 2);
        assert!(prefill_request.body.get("disagg_prefill_dp_rank").is_none());

        assert_eq!(
            decode_request.endpoint_url,
            "http://decode:30001/v1/completions"
        );
        assert_eq!(decode_request.body["data_parallel_rank"], 1);
        assert_eq!(decode_request.body["disagg_prefill_dp_rank"], 2);
        assert_eq!(decode_request.body["bootstrap_room"], 1234);
        assert!(matches!(prefill_request.body, Cow::Owned(_)));
        assert!(matches!(decode_request.body, Cow::Owned(_)));
    }

    #[tokio::test]
    async fn test_prepare_pd_worker_requests_preserves_non_dp_workers() {
        let prefill = BasicWorkerBuilder::new("http://prefill:30000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        let decode = BasicWorkerBuilder::new("http://decode:30001")
            .worker_type(WorkerType::Decode)
            .build();
        let request = json!({
            "prompt": "shared prefix",
            "max_tokens": 8,
            "bootstrap_room": 1234,
        });

        let (prefill_request, decode_request) =
            PDRouter::prepare_pd_worker_requests("/v1/completions", &request, &prefill, &decode)
                .await
                .unwrap();

        assert_eq!(
            prefill_request.endpoint_url,
            "http://prefill:30000/v1/completions"
        );
        assert_eq!(
            decode_request.endpoint_url,
            "http://decode:30001/v1/completions"
        );
        assert!(prefill_request.body.get("data_parallel_rank").is_none());
        assert!(decode_request.body.get("data_parallel_rank").is_none());
        assert!(decode_request.body.get("disagg_prefill_dp_rank").is_none());
        assert!(matches!(prefill_request.body, Cow::Borrowed(_)));
        assert!(matches!(decode_request.body, Cow::Borrowed(_)));
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let decode_guard = WorkerLoadGuard::new(decode_ref.clone(), None);
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                decode_ref.clone(),
                decode_guard,
            );

            // Prefill is a completed phase by the time a Decode stream is
            // returned; only Decode stays attached to the response body.
            assert_eq!(prefill_ref.load(), 0);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 0);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }

    #[test]
    fn test_prefill_admission_config_values() {
        assert_eq!(
            PrefillAdmissionConfig::from_values(None, Some("bad"), Some("bad")).unwrap(),
            None,
            "queue settings are inert when the room limiter is disabled"
        );
        assert_eq!(
            PrefillAdmissionConfig::from_values(Some("0"), None, None).unwrap(),
            None
        );

        let config = PrefillAdmissionConfig::from_values(Some("16"), Some("8192"), Some("21600"))
            .unwrap()
            .unwrap();
        assert_eq!(config.max_rooms_per_worker, 16);
        assert_eq!(config.queue_size_per_worker, 8192);
        assert_eq!(config.queue_timeout, Duration::from_secs(21_600));

        assert!(PrefillAdmissionConfig::from_values(Some("bad"), None, None).is_err());
        assert!(PrefillAdmissionConfig::from_values(Some("16"), None, Some("0")).is_err());
    }

    #[test]
    fn test_request_id_is_restored_for_prefill_and_decode_json() {
        let mut headers = HeaderMap::new();
        headers.insert(
            REQUEST_ID_HEADER,
            HeaderValue::from_static("pressure-full-2102-a1b2c3"),
        );
        let original = json!({"model": "qwen", "messages": []});

        let first =
            PDRouter::inject_request_id_into_value(original.clone(), Some(&headers), None).unwrap();
        let repeated =
            PDRouter::inject_request_id_into_value(original, Some(&headers), None).unwrap();

        assert_eq!(first[REQUEST_ID_BODY_KEY], "pressure-full-2102-a1b2c3");
        assert_eq!(first[REQUEST_ID_BODY_KEY], repeated[REQUEST_ID_BODY_KEY]);
    }

    #[test]
    fn test_request_id_batch_is_unique_and_deterministic() {
        let mut headers = HeaderMap::new();
        headers.insert(
            REQUEST_ID_HEADER,
            HeaderValue::from_static("pressure-batch"),
        );
        let value = PDRouter::inject_request_id_into_value(
            json!({"model": "qwen", "messages": []}),
            Some(&headers),
            Some(3),
        )
        .unwrap();

        assert_eq!(
            value[REQUEST_ID_BODY_KEY],
            json!(["pressure-batch-0", "pressure-batch-1", "pressure-batch-2"])
        );
    }

    #[test]
    fn test_invalid_request_id_is_not_injected() {
        for invalid in ["contains space", "slash/not/allowed", "", &"x".repeat(97)] {
            let mut headers = HeaderMap::new();
            headers.insert(REQUEST_ID_HEADER, HeaderValue::from_str(invalid).unwrap());
            let value = PDRouter::inject_request_id_into_value(
                json!({"model": "qwen", "messages": []}),
                Some(&headers),
                None,
            )
            .unwrap();
            assert!(
                value.get(REQUEST_ID_BODY_KEY).is_none(),
                "invalid={invalid:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_prefill_admission_queues_before_dispatch_and_releases_rooms() {
        use tokio::time::timeout;

        let admission = Arc::new(PrefillAdmission::new(Some(PrefillAdmissionConfig {
            max_rooms_per_worker: 2,
            queue_size_per_worker: 1,
            queue_timeout: Duration::from_secs(1),
        })));
        let worker = "http://prefill:30000";
        let first = admission.acquire(worker, 2).await.unwrap().unwrap();
        let state = admission.workers.get(worker).unwrap().clone();
        assert_eq!(state.active_rooms.load(Ordering::Acquire), 2);

        let waiting_admission = Arc::clone(&admission);
        let waiter =
            tokio::spawn(async move { waiting_admission.acquire("http://prefill:30000", 1).await });
        timeout(Duration::from_secs(1), async {
            while state.queued_requests.load(Ordering::Acquire) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("request should enter the Router-side queue");

        assert!(matches!(
            admission.acquire(worker, 1).await,
            Err(PrefillAdmissionError::QueueFull { .. })
        ));

        drop(first);
        let second = timeout(Duration::from_secs(1), waiter)
            .await
            .expect("queued request should receive the released room")
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(state.queued_requests.load(Ordering::Acquire), 0);
        assert_eq!(state.active_rooms.load(Ordering::Acquire), 1);
        drop(second);
        assert_eq!(state.active_rooms.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn test_router_queued_prefill_reservation_is_visible_to_cold_routing() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|index| {
                let worker = Arc::new(
                    BasicWorkerBuilder::new(format!("http://prefill-{index}:8000"))
                        .worker_type(WorkerType::Prefill {
                            bootstrap_port: Some(8998),
                        })
                        .build(),
                );
                worker.set_healthy(true);
                worker as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);

        let first = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("aaaa"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let _first_guard = WorkerLoadGuard::new(workers[first].clone(), None);
        let second = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("bbbb"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_ne!(first, second);
        let _second_guard = WorkerLoadGuard::new(workers[second].clone(), None);

        // Both workers now have one admitted/assigned request. A third cold
        // request may deterministically select either worker, but its routing
        // reservation must become visible while it waits for admission so the
        // fourth cold request selects the other worker instead of joining the
        // same invisible Router queue.
        let queued = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("cccc"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let _queued_guard = WorkerLoadGuard::new(workers[queued].clone(), None);
        let next = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("dddd"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_ne!(queued, next);
    }

    #[tokio::test]
    async fn test_prefill_admission_rejects_oversized_and_times_out() {
        let admission = PrefillAdmission::new(Some(PrefillAdmissionConfig {
            max_rooms_per_worker: 1,
            queue_size_per_worker: 1,
            queue_timeout: Duration::from_millis(10),
        }));
        let worker = "http://prefill:30000";

        assert!(matches!(
            admission.acquire(worker, 2).await,
            Err(PrefillAdmissionError::RequestTooLarge { .. })
        ));
        let first = admission.acquire(worker, 1).await.unwrap().unwrap();
        assert!(matches!(
            admission.acquire(worker, 1).await,
            Err(PrefillAdmissionError::Timeout { .. })
        ));
        drop(first);
    }
}

# DFLASH Mamba/GDN Replay Benchmark Report

日期：2026-06-11

分支：`feat/dflash-mamba-replay`

代码基线：`944bfc374 Fix DFLASH GDN replay with CUDA graph`

## 结论摘要

这轮已经完成一组完整的 replay 性能和显存 benchmark：`K in {0,1,4,8,16}`，并发 `C in {1,2,4,8,16}`，共 25 个 serving case，全部正常完成。

这里的 `K` 指 `--speculative-dflash-mamba-cache-steps`，也就是 target verify draft tokens 时缓存多少步 Mamba/GDN SSM intermediate states。`K=16` 等于 block size，是 full-cache baseline；`K<16` 会自动启用 replay；`K=0` 是 true zero-cache replay。

主要结果：

- 在固定 `mem_fraction_static=0.72` 下，`nvidia-smi` 看到的总显存占用接近不变，因为 SGLang 会把释放出来的中间状态预算转成更大的 KV/token capacity。
- `K=0` 将 target intermediate SSM cache 从 `19.12 GB/rank` 降到 `0.00 GB/rank`，target token capacity 从 `276,228` 提升到 `725,617`，约 `2.63x`。
- `K=4` 的内存/容量折中很好：intermediate SSM 为 `4.78 GB/rank`，target token capacity 为 `613,270`，约为 full-cache baseline 的 `2.22x`。
- 当前 partial-K (`K=1/4/8`) 的 TPS 只能作为诊断数据，不能作为最终性能结论，因为对应 accept length 已经偏离 `K=16` baseline。
- 正确性仍有 blocker：固定 prompt exact-output 对比中，`K=0` 与 `K=16` 在 CUDA graph 下精确一致，但 `K=1/4/8` 在 CUDA graph 下存在文本差异；`K=1` 关闭 CUDA graph 后与 `K=16` 精确一致。Serving 矩阵中的 mean accepted length 也显示 `K=1/4/8` 系统性低于 `K=0/16`，说明 partial-K replay 的剩余问题集中在 CUDA graph/post-verify state update 路径。

因此当前分支已经有比较明确的显存和性能证据，但 partial-K CUDA graph correctness 修复前，不建议直接作为最终 PR 合入。

## 当前参数语义

核心用户参数只有：

```text
--speculative-dflash-mamba-cache-steps K
```

含义：

- `K=None`：默认行为，缓存完整 verify block 的 Mamba/GDN SSM intermediate states。
- `K=block_size`：等价于 full-cache baseline，不启用 replay。
- `0 <= K < block_size`：自动启用 replay。
- `K=0`：不缓存 target verify draft tokens 的 SSM intermediate states，只保留 conv window intermediate cache，accepted tail 完全 replay。
- `K=1`：缓存第 1 个 verify step 的 SSM state，后续 accepted tail replay。
- `K>1`：缓存前 K 个 verify steps，超过 K 的 accepted tail replay。

内部仍有 `speculative_dflash_mamba_replay` 字段，但它不作为用户命令行参数暴露。当 `K < block_size` 时由参数 hook 自动打开，减少使用者负担。

本轮 benchmark 使用：

```text
--speculative-dflash-block-size 16
--speculative-dflash-mamba-cache-steps <K>
```

`--speculative-dflash-draft-window-size` 不是本报告的核心 replay 参数，本轮未启用。

## 测试环境

仓库：

```text
/mnt/cpfs/zyj/casual_spec/sglang-replay-upstream
```

Python 环境：

```text
/mnt/cpfs/zyj/casual_spec/sglang_env/bin/python
```

必须使用当前 repo 的 Python path，避免导入旧的 `/mnt/cpfs/zyj/casual_spec/sglang/python`：

```bash
export PYTHONPATH=/mnt/cpfs/zyj/casual_spec/sglang-replay-upstream/python:$PYTHONPATH
```

模型：

```text
Target: /mnt/cpfs/zyj/casual_spec/models/Qwen3.5-27B
Draft : /mnt/cpfs/zyj/casual_spec/models/Qwen3.5-27B-DFlash
```

这两个模型配置不应包含 Domino/projector 字段。

关键环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/cpfs/zyj/casual_spec/hf_home
export FLASHINFER_WORKSPACE_BASE=/tmp/zyj_flashinfer_workspace
export TVM_FFI_CACHE_DIR=/tmp/zyj_tvmffi_cache
export TRITON_CACHE_DIR=/tmp/zyj_triton_cache
export XDG_CACHE_HOME=/tmp/zyj_xdg_cache
export CUTE_DSL_CACHE_DIR=/tmp/zyj_cutedsl_cache
export NCCL_DEBUG=WARN
```

所有 server case 均带 `--disable-custom-all-reduce`。

## Benchmark 配置

服务端模板：

```bash
/mnt/cpfs/zyj/casual_spec/sglang_env/bin/python -m sglang.launch_server \
  --model-path /mnt/cpfs/zyj/casual_spec/models/Qwen3.5-27B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /mnt/cpfs/zyj/casual_spec/models/Qwen3.5-27B-DFlash \
  --speculative-dflash-block-size 16 \
  --speculative-dflash-mamba-cache-steps <K> \
  --tp-size 2 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.72 \
  --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 \
  --disable-radix-cache \
  --disable-custom-all-reduce
```

客户端模板：

```bash
/mnt/cpfs/zyj/casual_spec/sglang_env/bin/python -m sglang.bench_serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port 30000 \
  --model /mnt/cpfs/zyj/casual_spec/models/Qwen3.5-27B \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 64 \
  --random-range-ratio 1 \
  --num-prompts 80 \
  --max-concurrency <C> \
  --output-file <output>.jsonl \
  --extra-request-body '{"ignore_eos": true, "temperature": 0.0}'
```

矩阵：

```text
K: 0, 1, 4, 8, 16
C: 1, 2, 4, 8, 16
```

## 性能结果

注意：本节中 `K=1/4/8` 的性能结果是在 correctness blocker 尚未修复时采集的，只能用于定位和趋势参考。`K=0` 与 `K=16` 已通过 CUDA graph exact-output 对比；partial-K 的最终 TPS/latency 需要在 accept length 和 exact-output 对齐后重跑。

Output TPS：

| K/C | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 93.13 | 151.20 | 231.46 | 310.07 | 365.64 |
| 1 | 97.16 | 156.24 | 251.38 | 340.16 | 413.34 |
| 4 | 101.22 | 159.86 | 254.84 | 339.09 | 424.58 |
| 8 | 101.43 | 159.60 | 249.52 | 346.62 | 429.81 |
| 16 | 102.71 | 160.49 | 253.98 | 343.08 | 411.61 |

Mean TTFT, ms：

| K/C | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 109.67 | 134.63 | 153.16 | 194.91 | 436.81 |
| 1 | 101.08 | 134.36 | 156.97 | 171.70 | 310.94 |
| 4 | 101.00 | 133.23 | 147.05 | 191.34 | 307.18 |
| 8 | 100.52 | 130.70 | 147.85 | 187.94 | 310.67 |
| 16 | 97.80 | 130.93 | 143.31 | 168.78 | 318.56 |

Mean TPOT, ms：

| K/C | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 9.16 | 11.08 | 14.62 | 21.81 | 33.99 |
| 1 | 8.84 | 10.77 | 13.20 | 20.28 | 31.53 |
| 4 | 8.42 | 10.36 | 13.06 | 19.73 | 30.14 |
| 8 | 8.41 | 10.51 | 13.43 | 19.41 | 29.98 |
| 16 | 8.33 | 10.38 | 13.26 | 19.80 | 31.04 |

Mean E2E latency, ms：

| K/C | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 686.59 | 832.53 | 1074.04 | 1568.80 | 2578.02 |
| 1 | 658.08 | 812.82 | 988.87 | 1449.45 | 2297.43 |
| 4 | 631.62 | 785.82 | 969.58 | 1434.35 | 2206.28 |
| 8 | 630.34 | 792.75 | 993.68 | 1411.04 | 2199.26 |
| 16 | 622.50 | 784.73 | 978.70 | 1415.99 | 2274.27 |

Mean accepted length：

| K/C | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.82 | 3.80 | 3.77 | 3.76 | 3.75 |
| 1 | 3.47 | 3.46 | 3.48 | 3.48 | 3.48 |
| 4 | 3.70 | 3.66 | 3.65 | 3.65 | 3.62 |
| 8 | 3.71 | 3.70 | 3.67 | 3.68 | 3.69 |
| 16 | 3.82 | 3.80 | 3.77 | 3.76 | 3.75 |

这张表是 correctness blocker 的直接证据，不应解读为正常的 replay 性能差异。相同 prompt、相同 target/draft 模型下，`K` 只应该改变 target verify draft tokens 的 Mamba/GDN cache/replay 实现，不应该系统性改变接受长度。这里 `K=0` 与 `K=16` 基本对齐，而 `K=1/4/8` 全面偏低，说明 partial-cache replay 写回的 recurrent state 与 full-cache baseline 仍不等价。

观察：

- `K=0` 是显存最省的已验证配置，吞吐低于 `K=16`，原因是 accepted tail 的 SSM state 全量依赖 replay。
- `K=1/4/8` 在这轮中看起来有较高 TPS，但由于接受长度已经变小，不能作为有效加速结论。
- 修复 partial-K correctness 后，需要重跑同样矩阵，并确认 accepted length 与 `K=16` 对齐后再比较 TPS。

## 显存和容量结果

固定 `mem_fraction_static=0.72` 时，每个 K 的总 reserved memory 接近不变。更有意义的指标是中间状态 cache 与 token capacity。

| K | cached SSM steps | intermediate SSM GB/rank | intermediate conv GB/rank | target K GB/rank | target V GB/rank | draft K GB/rank | draft V GB/rank | target token capacity | gpu0 ready MB | gpu1 ready MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.00 | 0.37 | 11.07 | 11.07 | 3.46 | 3.46 | 725617 | 64904 | 64382 |
| 1 | 1 | 1.20 | 0.37 | 10.64 | 10.64 | 3.33 | 3.33 | 697531 | 64952 | 64430 |
| 4 | 4 | 4.78 | 0.37 | 9.36 | 9.36 | 2.92 | 2.92 | 613270 | 65180 | 64658 |
| 8 | 8 | 9.56 | 0.37 | 7.64 | 7.64 | 2.39 | 2.39 | 500923 | 65436 | 64914 |
| 16 | 16 | 19.12 | 0.37 | 4.21 | 4.21 | 1.32 | 1.32 | 276228 | 65456 | 64934 |

与 `K=16` full-cache baseline 相比：

| K | SSM intermediate saving GB/rank | target capacity ratio |
| --- | ---: | ---: |
| 0 | 19.12 | 2.63x |
| 1 | 17.92 | 2.53x |
| 4 | 14.34 | 2.22x |
| 8 | 9.56 | 1.81x |
| 16 | 0.00 | 1.00x |

这个结果符合 replay 的核心 motivation：linear attention/GDN 在 verify draft tokens 时，为每个 draft position 预留的 recurrent state snapshot 远大于 full attention 的单 token KV。减少或取消这些 draft-token recurrent state snapshots，可以把同样的静态显存预算转成更大的 KV/token capacity。

## 正确性结果

### 已通过

小规模 CUDA graph smoke：

- `K=0, C=2`：4/4 请求成功，output TPS `83.18`，mean E2E `754.68 ms`，TTFT `140.44 ms`，TPOT `19.81 ms`。
- `K=1, C=2`：4/4 请求成功，output TPS `114.46`，mean E2E `924.03 ms`，TTFT `137.79 ms`，TPOT `12.48 ms`。

固定 prompt exact-output：

- CUDA graph 下，`K=0` 与 `K=16` full-cache baseline 精确一致。
- 关闭 CUDA graph 后，`K=1` 与 `K=16` full-cache baseline 在 4 个固定 prompt 上精确一致。

### 未通过

固定 prompt exact-output，在 CUDA graph 开启时：

- `K=1`：4 个 prompt 中 3 个与 `K=16` 文本不同。
- `K=4`：4 个 prompt 中 3 个与 `K=16` 文本不同。
- `K=8`：4 个 prompt 中 2 个与 `K=16` 文本不同。

这些 case 没有 crash，属于输出文本差异。由于 `K=1` 在 eager 路径能与 baseline 精确一致，当前判断 partial-K replay 的数学路径大概率不是主要问题，剩余问题更可能在 CUDA graph 捕获和 replay metadata/state 更新路径。

## 已修问题

commit `944bfc374` 修复了一个 CUDA graph 下的 GDN replay metadata 问题：

- 增加 `GDNAttnBackend.init_forward_metadata_out_graph(...)`。
- CUDA graph replay 时按 graph batch size 选择 `_verify_replay_inputs_by_graph_bs[bs]`。
- 避免 K=0/C=2 CUDA graph 路径因 replay metadata 未正确设置导致 illegal memory access。

修复后：

- `git diff --check` 通过。
- `py_compile gdn_backend.py` 通过。
- `K=0/C=2` 和 `K=1/C=2` CUDA graph smoke 通过。

## 当前 blocker

PR 前需要继续修复：

1. partial-K CUDA graph exact-output mismatch。
2. 修复后重新运行固定 prompt exact-output，至少覆盖 `K=0/1/4/8/16`。
3. 修复后重新运行 serving benchmark 矩阵，确认性能没有回退。

建议优先排查：

- CUDA graph replay 时 GDN replay inputs 是否按 graph batch size 与当前 request layout 完全一致。
- `accepted_steps`、`accepted_lengths`、commit lengths 在 graph replay 中是否被正确复用或覆盖。
- partial cached SSM scatter 后，再 replay uncached tail 的 source/destination state indices 是否在 graph 路径中使用了 stale tensor。
- `K=0` graph 正确而 `K=1/4/8` graph 不正确，说明 zero-cache path 与 partial-cache path 的差异是优先排查范围。

### 2026-06-11 follow-up

对照旧工作分支 `dev/dflash-mamba-replay-study` 后确认，旧分支确实有一系列后续 GDN replay 修复和优化，包括：

```text
93338136e Support source-destination GDN replay
db08467a9 Use sequence lengths for GDN replay
df9a584f6 Use dense K1 GDN replay
83ba1495c Reuse K1 GDN replay tail lengths
8c8b5dab9 Guard K0 GDN replay length reuse
```

当前 upstream replay 分支只迁移了 replay 主线和一个 CUDA graph metadata 修复，没有完整迁移旧分支后续 patch。旧分支报告里记录过 accept length unchanged，因此当前 `K=1/4/8` accept length 下降不是预期行为。

已尝试最小迁移旧分支的 dense replay sequence-length 寻址：

- 在 `fused_sigmoid_gating_recurrent.py` 中允许 `input_sequence_lengths + input_token_stride` 做物理位置寻址。
- 在 `gdn_backend.py` dense replay 路径中使用 `req_indices=None`。

复测路径：

```text
/mnt/cpfs/zyj/casual_spec/bench_outputs/dflash_mamba_replay_20260611/correctness_seq_len_patch_k1
```

结果：`K=1` 与 `K=16` 仍未 exact-output 对齐，4/4 prompt mismatch；部分 prompt 的 `spec_accept_length` 也继续偏离。因此该最小迁移不足以解决当前 upstream 基线上的 partial-K CUDA graph/post-verify state update 问题，未提交代码改动。

## Artifact 路径

完整 benchmark：

```text
/mnt/cpfs/zyj/casual_spec/bench_outputs/dflash_mamba_replay_20260611/matrix_full_944bfc374
```

关键文件：

```text
summary.csv
memory.csv
k0/bench_k0_c*.jsonl
k1/bench_k1_c*.jsonl
k4/bench_k4_c*.jsonl
k8/bench_k8_c*.jsonl
k16/bench_k16_c*.jsonl
```

正确性：

```text
/mnt/cpfs/zyj/casual_spec/bench_outputs/dflash_mamba_replay_20260611/correctness_944bfc374
/mnt/cpfs/zyj/casual_spec/bench_outputs/dflash_mamba_replay_20260611/correctness_k1_eager_944bfc374
```

## PR 准备状态

当前状态：

- replay memory control feature 已实现。
- zero-cache replay (`K=0`) 在 CUDA graph exact-output 对比中已经通过。
- full benchmark matrix 已完成，有显存和 TPS 数据。
- partial-K CUDA graph correctness 尚未通过。

建议 PR 策略：

1. 先修 partial-K CUDA graph correctness。
2. 若短期只想提最小 PR，可以考虑只开放 `K=0` 和 `K=block_size`，暂时禁用 `0<K<block_size` 的 CUDA graph 路径；但这会牺牲 `K=4/8` 这类性能和显存更均衡的配置。
3. 更完整的 PR 应支持 `K=0/1/4/8/16` 全部 exact-output 通过，并附上本报告中的 benchmark 表格。

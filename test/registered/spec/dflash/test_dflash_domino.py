import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "Huang2020/Qwen3-8B-Domino-b16"


register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")


class TestDFlashDominoServer(GSM8KMixin, CustomTestCase):
    model = TARGET_MODEL
    gsm8k_num_questions = 200
    gsm8k_accuracy_thres = 0.90
    gsm8k_accept_length_thres = 4.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
            envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True),
            envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "triton",
                    "--speculative-draft-attention-backend",
                    "triton",
                    "--speculative-algorithm",
                    "DFLASH",
                    "--speculative-draft-model-path",
                    DRAFT_MODEL,
                    "--dtype",
                    "bfloat16",
                    "--page-size",
                    "1",
                    "--max-running-requests",
                    "64",
                    "--mem-fraction-static",
                    "0.7",
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()

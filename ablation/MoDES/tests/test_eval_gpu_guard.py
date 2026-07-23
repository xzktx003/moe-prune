import unittest

from utils import ensure_single_gpu_evaluation


class EvalGpuGuardTest(unittest.TestCase):
    def test_defaults_to_gpu_7_when_env_missing(self):
        env = {}

        selected = ensure_single_gpu_evaluation(env=env)

        self.assertEqual(selected, "7")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")

    def test_rejects_multi_gpu_visible_devices(self):
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}

        with self.assertRaisesRegex(RuntimeError, "exactly one GPU"):
            ensure_single_gpu_evaluation(env=env)

    def test_rejects_multi_process_world_size(self):
        env = {"CUDA_VISIBLE_DEVICES": "7", "WORLD_SIZE": "2"}

        with self.assertRaisesRegex(RuntimeError, "Multi-process evaluation"):
            ensure_single_gpu_evaluation(env=env)


if __name__ == "__main__":
    unittest.main()

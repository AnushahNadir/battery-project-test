import unittest

from src.modeling import run_full_pipeline


class TestPipelineEntrypoint(unittest.TestCase):
    def test_run_full_pipeline_has_main(self):
        self.assertTrue(hasattr(run_full_pipeline, "main"))
        self.assertTrue(callable(run_full_pipeline.main))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from mini_office.benchmark import run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_end_to_end_stream(self):
        result = run_benchmark()
        self.assertEqual(result.tasks, 24)
        self.assertEqual(result.successful_tasks, 24)
        self.assertEqual(result.workflow_tasks, 12)
        self.assertEqual(result.model_free_workflow_tasks, 9)
        self.assertEqual(result.slm_workflow_tasks, 3)
        self.assertEqual(result.learned_workflows, {"active": 4})
        self.assertEqual(result.false_workflow_offers, 0)
        self.assertGreater(result.token_reduction_rate, 0.4)


if __name__ == "__main__":
    unittest.main()

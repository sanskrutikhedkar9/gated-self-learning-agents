from __future__ import annotations

import unittest

from mini_office.agent import ReferenceAgent
from mini_office.dataset import load_tasks
from mini_office.world import MiniOfficeWorld
from self_learning_flows.background import BackgroundWorkflowLearner
from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.models import WorkflowStatus


class BackgroundLearnerTests(unittest.TestCase):
    def test_always_on_observer_learns_off_hot_path(self):
        store = InMemoryStore()
        learner = BackgroundWorkflowLearner(SelfLearningFlowEngine(store))
        learner.start()
        tasks = [task for task in load_tasks() if task.pattern == "onboard_employee"]
        for task in tasks[:3]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            learner.submit(ReferenceAgent(world).run(task))
        learner.flush()
        learner.stop()
        self.assertEqual(learner.errors, [])
        workflows = store.list_workflows()
        self.assertEqual(len(workflows), 1)
        self.assertEqual(workflows[0].status, WorkflowStatus.SHADOW)


if __name__ == "__main__":
    unittest.main()

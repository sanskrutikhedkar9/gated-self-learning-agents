"""Optional event-driven worker for moving synthesis off the task hot path."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from .engine import SelfLearningFlowEngine
from .models import TaskEpisode


@dataclass(slots=True)
class LearnerEvent:
    episode: TaskEpisode | None = None
    stop: bool = False


class BackgroundWorkflowLearner:
    def __init__(self, engine: SelfLearningFlowEngine, *, max_queue_size: int = 10_000):
        self.engine = engine
        self.queue: queue.Queue[LearnerEvent] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self.errors: list[Exception] = []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="workflow-learner", daemon=True)
        self._thread.start()

    def submit(self, episode: TaskEpisode, *, timeout: float = 1.0) -> None:
        self.queue.put(LearnerEvent(episode=episode), timeout=timeout)

    def flush(self) -> None:
        self.queue.join()

    def stop(self, *, timeout: float = 10.0) -> None:
        if not self._thread:
            return
        self.queue.put(LearnerEvent(stop=True))
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            event = self.queue.get()
            try:
                if event.stop:
                    return
                if event.episode is not None:
                    self.engine.observe(event.episode)
            except Exception as exc:  # expose background errors without killing the observer
                self.errors.append(exc)
            finally:
                self.queue.task_done()

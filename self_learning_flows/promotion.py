"""Evidence-based workflow lifecycle policy."""

from __future__ import annotations

from dataclasses import dataclass

from .models import WorkflowDefinition, WorkflowStatus
from .scoring import wilson_lower_bound


@dataclass(slots=True)
class PromotionConfig:
    min_pattern_observations: int = 3
    min_shadow_executions: int = 3
    min_success_rate: float = 0.90
    min_wilson_lower_bound: float = 0.40
    quarantine_after_consecutive_failures: int = 2


class PromotionPolicy:
    def __init__(self, config: PromotionConfig | None = None):
        self.config = config or PromotionConfig()

    def evaluate(self, workflow: WorkflowDefinition) -> WorkflowStatus:
        stats = workflow.stats
        if workflow.status in {WorkflowStatus.QUARANTINED, WorkflowStatus.RETIRED}:
            return workflow.status
        if stats.consecutive_failures >= self.config.quarantine_after_consecutive_failures:
            return WorkflowStatus.QUARANTINED
        if (
            workflow.status == WorkflowStatus.CANDIDATE
            and stats.pattern_observations >= self.config.min_pattern_observations
        ):
            return WorkflowStatus.SHADOW
        if (
            workflow.status == WorkflowStatus.SHADOW
            and stats.executions >= self.config.min_shadow_executions
        ):
            observed_rate = stats.successes / stats.executions if stats.executions else 0.0
            lower = wilson_lower_bound(stats.successes, stats.executions)
            if (
                observed_rate >= self.config.min_success_rate
                and lower >= self.config.min_wilson_lower_bound
            ):
                return WorkflowStatus.ACTIVE
        return workflow.status


class ResearchPromotionPolicy(PromotionPolicy):
    """Stricter defaults intended for reported experiments or deployment."""

    def __init__(self):
        super().__init__(
            PromotionConfig(
                min_pattern_observations=5,
                min_shadow_executions=10,
                min_success_rate=0.95,
                min_wilson_lower_bound=0.75,
                quarantine_after_consecutive_failures=1,
            )
        )

"""Predeclared experiment phases and immutable test-time workflow snapshots."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .models import ConfirmationDecision, ExecutionResult, TaskEpisode, WorkflowProposal


class ExperimentPhase(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class ProtocolViolation(RuntimeError):
    """Raised before an operation that would contaminate an experiment phase."""


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    name: str = "appworld-continual-v1"
    train_splits: tuple[str, ...] = ("train",)
    dev_splits: tuple[str, ...] = ("dev",)
    test_splits: tuple[str, ...] = ("test_normal", "test_challenge")
    learn_on_train: bool = True
    calibrate_on_dev: bool = True
    freeze_before_test: bool = True
    fresh_world_fallback: bool = True
    supported_step_kinds: tuple[str, ...] = (
        "tool",
        "compute",
        "paginate",
        "foreach",
        "filter",
        "reduce",
        "branch",
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtocolConfig:
        known = {
            "name",
            "train_splits",
            "dev_splits",
            "test_splits",
            "learn_on_train",
            "calibrate_on_dev",
            "freeze_before_test",
            "fresh_world_fallback",
            "supported_step_kinds",
            "metadata",
        }
        unexpected = set(payload) - known
        if unexpected:
            raise ValueError(f"Unknown protocol fields: {', '.join(sorted(unexpected))}")
        converted = dict(payload)
        for key in ("train_splits", "dev_splits", "test_splits", "supported_step_kinds"):
            if key in converted:
                converted[key] = tuple(converted[key])
        return cls(**converted)

    @classmethod
    def load(cls, path: str | Path) -> ProtocolConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Protocol config must be a JSON object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train_splits": list(self.train_splits),
            "dev_splits": list(self.dev_splits),
            "test_splits": list(self.test_splits),
            "learn_on_train": self.learn_on_train,
            "calibrate_on_dev": self.calibrate_on_dev,
            "freeze_before_test": self.freeze_before_test,
            "fresh_world_fallback": self.fresh_world_fallback,
            "supported_step_kinds": list(self.supported_step_kinds),
            "metadata": self.metadata,
        }


class ProtocolGuard:
    """Owns all phase-dependent mutations used by benchmark runners."""

    MANIFEST_SCHEMA_VERSION = 1

    def __init__(self, config: ProtocolConfig, phase: ExperimentPhase, split: str):
        self.config = config
        self.phase = phase
        self.split = split
        self._validate_split()

    @property
    def allows_learning(self) -> bool:
        return self.phase == ExperimentPhase.TRAIN and self.config.learn_on_train

    @property
    def allows_calibration(self) -> bool:
        return self.phase in {ExperimentPhase.TRAIN, ExperimentPhase.DEV} and (
            self.phase != ExperimentPhase.DEV or self.config.calibrate_on_dev
        )

    def observe(self, engine: Any, episode: TaskEpisode):
        if not self.allows_learning:
            raise ProtocolViolation(f"Workflow compilation is disabled during {self.phase.value}")
        return engine.observe(episode)

    def record_execution(
        self,
        engine: Any,
        workflow_id: str,
        result: ExecutionResult,
    ) -> Any | None:
        if not self.allows_calibration:
            return None
        return engine.record_execution(workflow_id, result)

    @staticmethod
    def approve_in_memory(proposal: WorkflowProposal) -> None:
        """Bind an exact version and variables without persisting task/test feedback."""
        proposal.confirmation_decision = ConfirmationDecision.APPROVE
        proposal.confirmed_variables = dict(proposal.variables)

    def seal(
        self,
        store: Any,
        path: str | Path,
        *,
        source_commit: str,
        run_config: dict[str, Any],
    ) -> dict[str, Any]:
        if self.phase == ExperimentPhase.TEST:
            raise ProtocolViolation("A workflow snapshot must be sealed before test")
        if not source_commit or source_commit == "unknown":
            raise ProtocolViolation("A real source commit is required to seal an experiment")
        workflows = self._workflow_payloads(store)
        manifest = {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "protocol": self.config.to_dict(),
            "protocol_digest": self._digest(self.config.to_dict()),
            "sealed_after_phase": self.phase.value,
            "sealed_at": time.time(),
            "source_commit": source_commit,
            "workflow_count": len(workflows),
            "program_digest": self._program_digest(workflows),
            "state_digest": self._digest(workflows),
            "workflows": [
                {
                    "workflow_id": item["workflow_id"],
                    "version": item["version"],
                    "status": item["status"],
                    "structural_signature": item["structural_signature"],
                }
                for item in workflows
            ],
            "run_config": run_config,
            "environment": self.environment_manifest(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return manifest

    def validate_frozen(
        self,
        store: Any,
        path: str | Path,
        *,
        expected_source_commit: str | None = None,
    ) -> dict[str, Any]:
        if self.phase != ExperimentPhase.TEST:
            raise ProtocolViolation("Frozen manifests are only consumed during test")
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != self.MANIFEST_SCHEMA_VERSION:
            raise ProtocolViolation("Unsupported freeze manifest schema")
        if manifest.get("protocol_digest") != self._digest(self.config.to_dict()):
            raise ProtocolViolation("Protocol config differs from the sealed protocol")
        if (
            expected_source_commit is not None
            and manifest.get("source_commit") != expected_source_commit
        ):
            raise ProtocolViolation("Source commit differs from the sealed experiment")
        workflows = self._workflow_payloads(store)
        if manifest.get("program_digest") != self._program_digest(workflows):
            raise ProtocolViolation("Executable workflow definitions changed after freezing")
        if manifest.get("state_digest") != self._digest(workflows):
            raise ProtocolViolation(
                "Workflow state or calibration statistics changed after freezing"
            )
        return manifest

    def _validate_split(self) -> None:
        allowed = {
            ExperimentPhase.TRAIN: self.config.train_splits,
            ExperimentPhase.DEV: self.config.dev_splits,
            ExperimentPhase.TEST: self.config.test_splits,
        }[self.phase]
        if self.split not in allowed:
            raise ProtocolViolation(
                f"Split {self.split!r} is not declared for phase {self.phase.value}; "
                f"expected one of {', '.join(allowed)}"
            )

    @staticmethod
    def _workflow_payloads(store: Any) -> list[dict[str, Any]]:
        return sorted(
            (workflow.to_dict() for workflow in store.list_workflows()),
            key=lambda item: (item["workflow_id"], item["version"]),
        )

    @classmethod
    def _program_digest(cls, workflows: list[dict[str, Any]]) -> str:
        fields = (
            "workflow_id",
            "version",
            "scope",
            "variables",
            "steps",
            "structural_signature",
            "preconditions",
            "postconditions",
            "tool_schema_hashes",
            "required_tools",
            "environment",
        )
        programs = [{key: item.get(key) for key in fields} for item in workflows]
        return cls._digest(programs)

    @staticmethod
    def _digest(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def environment_manifest() -> dict[str, Any]:
        packages: dict[str, str] = {}
        for package in ("self-learning-flows", "appworld", "openai"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                continue
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
        }

"""Public API for SelfLearningFlows."""

from .compiler import EvidenceGatedWorkflowCompiler, ModelAssistedWorkflowCompiler
from .compute import FunctionComputeBackend, TieredComputeBackend
from .engine import InMemoryStore, SelfLearningFlowEngine
from .execution import ToolRegistry, WorkflowExecutor
from .extraction import StructuredVariableExtractor
from .models import (
    ComputationKind,
    ConfirmationDecision,
    ExecutionResult,
    TaskEpisode,
    TaskRequest,
    ToolCallTrace,
    WorkflowDefinition,
    WorkflowMatch,
    WorkflowProposal,
    WorkflowStatus,
)
from .research_protocol import (
    ExperimentPhase,
    ProtocolConfig,
    ProtocolGuard,
    ProtocolViolation,
)
from .storage import SQLiteStore

__all__ = [
    "ComputationKind",
    "ConfirmationDecision",
    "ExecutionResult",
    "ExperimentPhase",
    "EvidenceGatedWorkflowCompiler",
    "SelfLearningFlowEngine",
    "InMemoryStore",
    "ModelAssistedWorkflowCompiler",
    "ProtocolConfig",
    "ProtocolGuard",
    "ProtocolViolation",
    "FunctionComputeBackend",
    "TieredComputeBackend",
    "SQLiteStore",
    "StructuredVariableExtractor",
    "TaskEpisode",
    "TaskRequest",
    "ToolCallTrace",
    "ToolRegistry",
    "WorkflowDefinition",
    "WorkflowExecutor",
    "WorkflowMatch",
    "WorkflowProposal",
    "WorkflowStatus",
]

__version__ = "0.1.0"

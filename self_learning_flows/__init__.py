"""Public API for SelfLearningFlows."""

from .compiler import EvidenceGatedWorkflowCompiler, ModelAssistedWorkflowCompiler
from .compute import FunctionComputeBackend, TieredComputeBackend
from .discovery import HybridEpisodeFamilyDiscoverer
from .engine import InMemoryStore, SelfLearningFlowEngine
from .execution import ToolRegistry, WorkflowExecutor
from .extraction import StructuredVariableExtractor
from .matching import ModelSemanticReranker
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
from .synthesis import (
    RecordedTraceReplayValidator,
    ValidatedWorkflowCompiler,
    WorkflowProgramValidator,
    WorkflowSynthesisError,
    WorkflowValidationReport,
)

__all__ = [
    "ComputationKind",
    "ConfirmationDecision",
    "ExecutionResult",
    "ExperimentPhase",
    "EvidenceGatedWorkflowCompiler",
    "HybridEpisodeFamilyDiscoverer",
    "SelfLearningFlowEngine",
    "InMemoryStore",
    "ModelAssistedWorkflowCompiler",
    "ModelSemanticReranker",
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
    "WorkflowProgramValidator",
    "WorkflowSynthesisError",
    "WorkflowValidationReport",
    "ValidatedWorkflowCompiler",
    "RecordedTraceReplayValidator",
    "WorkflowStatus",
]

__version__ = "0.2.0"

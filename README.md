# SelfLearningFlows

SelfLearningFlows is a framework-neutral procedural-memory layer for tool-using
agents. It watches verified successful executions, finds repeated tool paths,
compiles them into typed workflows, asks for confirmation on a matching future
task, and executes each step with the cheapest safe computation tier.

The intended behavior is simple:

```text
unfamiliar task -> full agent -> verified trace -> candidate workflow
repeated trace  -> shadow workflow -> confirmed replay -> active workflow
familiar task   -> tools only / small model -> verify -> done or safe fallback
```

This repository contains a working library, a LangGraph adapter, an AppWorld
baseline/treatment runner, a stateful Mini Office environment, tests, and a
versioned 24-task continual-learning dataset.

## What works now

- Always-on, asynchronous observation of every task; only externally verified
  successes can become procedural memory.
- Structural discovery from normalized multi-agent tool traces.
- Deterministic anti-unification of concrete values into `$input.*` and
  `$steps.*` data-flow bindings.
- A bounded workflow IR for pagination, foreach loops, deterministic filters
  and reducers, and explicit branches. Bound exhaustion fails rather than
  silently truncating work.
- Optional LLM annotation using strict structured output. The model cannot
  invent, delete, or reorder tools in the conservative compiler.
- Four execution tiers: deterministic, SLM, LLM, and full-agent fallback.
- Confirmation payloads with workflow name, description, variables, confidence,
  and approve/edit/reject/full-agent choices.
- Candidate -> shadow -> active -> quarantined lifecycle based on observed
  support, verified execution rate, Wilson confidence, and consecutive failures.
- Tool-contract hashing, postcondition verification, immutable episodes,
  versioned workflows, SQLite WAL persistence, and negative feedback memory.
- Core package has no mandatory third-party dependency.

## Reproduce the current result

Python 3.10+ is sufficient for the core benchmark:

```bash
python -m self_learning_flows benchmark
python experiments/run_multiseed.py
python -m unittest discover -s tests -v
python -m self_learning_flows demo
```

The first command writes `experiments/results/mini_office_v1.json`. On the
checked-in task order, the current implementation produces:

| Metric | Result |
|---|---:|
| Environment-verified success | 24 / 24 |
| Full-agent routes | 12 / 24 |
| Learned-workflow routes | 12 / 24 |
| Fully model-free workflow routes | 9 / 12 |
| SLM workflow routes | 3 / 12 |
| Workflow fallbacks | 0 |
| False offers on six negative probes | 0 |
| Active workflows at end | 4 |

Across five shuffled chronological orders, verified success remains 100%; the
stricter intent/ambiguity gate avoids 46.7% +/- 1.9% of full-agent calls and the
reference token reduction is 43.8% +/- 1.8%.

The APIs and state verifiers are actually executed. Token counts in this first
offline benchmark use a declared reference cost model, not provider billing;
they are useful for regression tests but are not paper evidence. Provider-backed
and external-benchmark evaluation is listed in the research roadmap.

## Use it from any agent framework

Normalize a completed run into `TaskEpisode`, then observe it:

```python
from self_learning_flows import SQLiteStore, SelfLearningFlowEngine

engine = SelfLearningFlowEngine(SQLiteStore("state/workflows.db"))
workflow = engine.observe(verified_episode)
```

Before a future task, request a proposal:

```python
proposal = engine.propose(task_request)
if proposal:
    show_to_user(proposal.confirmation_payload())
```

After approval, execute against your own registered tools:

```python
result = engine.execute(task_request, proposal, workflow_executor)
if result.escalated:
    run_existing_full_agent(task_request)
```

Framework adapters only normalize state and route control. The discovery,
promotion, retrieval, and execution code does not import LangGraph, ADK, or
another orchestration framework.

## LangGraph integration

Install the optional adapter and use a checkpointer so confirmation interrupts
can resume with the same thread ID:

```bash
pip install -e ".[langgraph]"
```

`self_learning_flows.adapters.langgraph.build_self_learning_graph` adds a gate before an
existing full-agent node. See `examples/langgraph_self_learning.py`. LangGraph's
checkpointer remains short-term thread state; SelfLearningFlows is the learned,
executable procedural-memory layer above it.

## Optional models

- No model: deterministic tool paths, parsers, templates, rules.
- SLM: bounded classification, extraction, or short transformation through an
  OpenAI-compatible local endpoint.
- LLM: conservative workflow annotation or genuinely difficult semantic steps.
- Full agent: unmatched tasks, rejected proposals, failed verification, or
  unsafe/ambiguous conditions.

`ModelAssistedWorkflowCompiler` works with `AnthropicStructuredModel`; the
provider uses strict JSON schema output. The original prototype's misspelled
`ANTROPIC_API_KEY` and the correct `ANTHROPIC_API_KEY` are both supported.

```python
from self_learning_flows import (
    EvidenceGatedWorkflowCompiler,
    ModelAssistedWorkflowCompiler,
    SelfLearningFlowEngine,
    SQLiteStore,
    StructuredVariableExtractor,
)
from self_learning_flows.providers import AnthropicStructuredModel

model = AnthropicStructuredModel.from_env()
engine = SelfLearningFlowEngine(
    SQLiteStore("state/workflows.db"),
    compiler=EvidenceGatedWorkflowCompiler(ModelAssistedWorkflowCompiler(model)),
    variable_extractor=StructuredVariableExtractor(model),
)
```

The evidence gate keeps the paid compiler off one-off tasks; it is invoked only
after the configured recurrence threshold. Extraction can instead use a local
SLM through `OpenAICompatibleStructuredModel`.

## Repository map

```text
self_learning_flows/      framework-neutral package
  adapters/               LangGraph and AppWorld boundaries
mini_office/              executable multi-agent office sandbox and versioned task stream
experiments/              runners and committed result artifacts
tests/                    unit, lifecycle, storage, and end-to-end tests
docs/                     architecture and research positioning
```

Read [the architecture](docs/architecture.md) and
[research positioning](docs/research.md) before extending the system. Before an
AppWorld experiment, follow the locked, phase-safe
[AppWorld runbook](experiments/appworld/README.md) and require a green preflight.
The runner is implemented, but no external benchmark result is claimed until
the paid runs and official evaluations are completed.

## Research status

The broad idea “learn and retrieve workflows from memory” is not novel by
itself. A defensible paper contribution is the combination evaluated here:

1. continual trace-to-executable-workflow induction rather than prompt-only
   memory;
2. empirical, per-step **progressive model shedding** from full reasoning to
   LLM, SLM, or model-free execution;
3. risk-controlled promotion, confirmation, abstention, verification,
   quarantine, and schema-drift handling;
4. framework-neutral portability across single- and multi-agent runtimes.

That is a hypothesis and an implemented research substrate, not yet a novelty
claim. A paper still needs strong external baselines, multi-seed experiments,
failure analysis, and ablations described in `docs/research.md`.

## License

MIT.

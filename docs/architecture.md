# Architecture

## Online path

```mermaid
flowchart LR
    U[Task request] --> M{Eligible match?}
    M -- no --> A[Existing full agent]
    M -- yes --> C{User confirms?}
    C -- reject or full agent --> A
    C -- approve or edit --> W[Typed workflow executor]
    W --> D[Tool/API only]
    W --> S[Bounded SLM]
    W --> L[LLM only where required]
    D --> V{State verifier}
    S --> V
    L --> V
    V -- observable pass --> O[Return result + update evidence]
    V -- execution failure --> Q[Record failure / quarantine]
    Q --> R[Restore clean task state]
    R --> A
    A --> V2{External verifier}
    V2 -- pass --> E[Immutable successful episode]
    E --> P[Hybrid task-family discovery]
    P --> S2[Constrained LLM program proposal]
    S2 --> G[Static validation + recorded replay]
    G --> K[Candidate / shadow challenger]
```

Observation is cheap and always on. `BackgroundWorkflowLearner` puts normalized
episodes on a queue. Expensive workflow synthesis only happens after a verified
success and can be placed behind a recurrence threshold in a production worker.

## Learning pipeline

The learning path separates semantic judgment from execution authority:

```text
verified traces
  -> exact/hybrid family discovery
  -> symbolic evidence (no raw results or concrete argument values)
  -> strict-JSON LLM proposal in the bounded workflow IR
  -> local capability/reference/schema validation
  -> offline replay against every source episode
  -> bounded repair (maximum two by default)
  -> candidate/shadow workflow
```

Exact signatures require no model. Uncertain family assignments are reviewed by
a model only after hard scope, action, and side-effect gates. A low-confidence or
failed review creates a separate family. At request time, lexical retrieval
shortlists workflows and a semantic reranker handles paraphrases; it cannot
override status, scope, environment, tool availability, or negation checks.

The compiler receives instructions with observed argument values templated out,
symbolic `$input`/`$evidence` data flow, result shapes, and tool schemas. Concrete
literals remain in process and can only be selected through opaque evidence IDs.
The model never receives a callable tool registry.

Recorded replay substitutes saved tool results and requires the proposal to
reproduce each source episode's operation order and arguments. This validates
observed paths, not unseen branches; unseen control flow remains a stated threat
and must earn evidence through shadow executions.

## Data contracts

- `TaskEpisode`: instruction, scope, variables, outcome, verification status,
  cost metadata, and normalized tool calls.
- `WorkflowDefinition`: typed variables, ordered steps, data-flow references,
  required tool contracts, examples, negative examples, version, and statistics.
- `WorkflowProposal`: workflow, match decomposition, extracted values, missing
  values, and a serializable confirmation payload.
- `ExecutionResult`: actual step traces, executor tiers, verifier result, cost,
  error, and escalation flag.

Values such as `$input.new_start` and `$steps.find_event.id` make learned data
flow explicit and auditable. Tool calls are deterministic once arguments are
bound. Only `compute.*` steps may use a model.

Execution fails closed before invoking tools when there is no outcome verifier,
when a required tool contract has no learned schema hash, or when the current
schema hash differs. A failed workflow reports whether side effects may already
have occurred; callers must restore a clean task state before full-agent
fallback.

The workflow IR is deliberately restricted instead of embedding arbitrary
generated Python. In addition to tool and bounded-compute steps it supports:

- pagination with an explicit page argument, stop signal, and maximum bound;
- foreach over a materialized collection with a maximum item count;
- filters using an allow-listed predicate DSL;
- deterministic reducers such as unique, count, sort, and top-k;
- branches with explicit then/else subprograms.

Nested steps retain tool-schema contracts and globally unique step IDs. A loop
that reaches its bound without a stop signal fails closed, making incomplete
work visible rather than returning a plausible partial result.

## AppWorld fallback and evaluator isolation

The treatment runner uses two verification boundaries. Observable completion
(`task_completed`) may trigger a clean-world fallback when execution itself
fails. The official AppWorld evaluator is called only after a completed attempt
and supplies the reported score. An official-evaluator failure never selects a
second attempt, so hidden benchmark feedback cannot become a test-time oracle.

Fallback destroys the attempted world and constructs a new `AppWorld` for the
same task and experiment. Reinitialization restores the input DB and also
replaces the requester, API wrappers, Python namespace, counters, and frozen
clock. Tests assert that the workflow and fallback agents receive distinct
world identities.

## Research phase state machine

`ProtocolGuard` is the only benchmark-facing mutation boundary:

```text
train: compile + calibrate -> dev: calibrate only -> seal -> test: read-only
```

The seal hashes both executable programs and full workflow state. Test validates
the protocol, source commit, program digest, and state digest before and after
execution. Test evaluator outcomes and execution statistics are not written
back to procedural memory.

## Two independent confidence signals

Pattern support answers “have we seen the same procedure repeatedly?” Execution
reliability answers “does the compiled procedure work on new inputs?” They are
not combined into one misleading count:

- three verified source episodes promote a candidate to shadow by default;
- three confirmed, verified shadow executions at >=90% observed success and an
  adequate Wilson lower bound promote it to active;
- two consecutive failures quarantine it;
- `ResearchPromotionPolicy` uses materially stricter thresholds.

## Computation routing

The compiler assigns the minimum observed/validated tier per compute step:

```text
deterministic function -> SLM -> LLM -> full agent
```

`fallback_executors` supplies the upward safety ladder. Downward movement is
allowed only through a new workflow version and shadow evaluation. This avoids
silently replacing a capable model based on one lucky example.

## Incumbent/challenger updates

Candidate workflows may be refined while still unpublished. Once a workflow is
shadow or active, structurally novel family evidence creates a separate
challenger instead of rewriting it. Train/dev routing deliberately exercises a
shadow challenger. Test routing sees active workflows only. When the challenger
meets the same promotion requirements, it becomes active and retires the prior
incumbent; failed challengers leave the incumbent untouched.

## Scaling boundary

SQLite WAL and the background thread are the local reference deployment. The
core depends on `WorkflowStore` and `ComputeBackend` protocols, so production
systems can use a transactional database, a durable event stream, and separate
compiler workers without changing routing semantics. Episode IDs make ingestion
idempotent; workflow versions make updates auditable.

The next production store should use PostgreSQL with row/version locking, while
high-volume episode ingestion should use Kafka, Pub/Sub, or the application's
existing queue. A vector index is optional: exact structural signatures and
typed contracts remain the first-stage safety filter.

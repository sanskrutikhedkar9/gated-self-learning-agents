# Research positioning and experiment plan

## What prior work already establishes

- [Reflexion](https://arxiv.org/abs/2303.11366) stores linguistic reflection in
  episodic memory to improve later decisions. It still uses the language agent
  to reason from that memory.
- [ExpeL](https://arxiv.org/abs/2308.10144) extracts natural-language insights
  from experience and retrieves them at inference time without weight updates.
- [Voyager](https://arxiv.org/abs/2305.16291) learns an executable skill library
  and self-verifies code in Minecraft. It demonstrates continual executable
  skills but is domain-specific and still uses GPT-4 for curriculum/planning.
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) is the closest early
  baseline: it induces reusable workflows offline or online, retrieves them,
  and provides them to an agent to guide later generation.
- [FlowBench](https://aclanthology.org/2024.findings-emnlp.638/) shows that the
  representation of workflow knowledge materially affects LLM planning and that
  current agents remain weak at workflow-guided planning.
- [LEGOMem](https://arxiv.org/abs/2510.04851) decomposes multi-agent trajectories
  into full-task and agent-local procedural memories. Its retrieved memories
  guide LLM orchestration and tool use; it also studies smaller-model teams.
- [AFTER](https://arxiv.org/abs/2606.23127) is especially important for the next
  evaluation: it reports that procedural memories can over-specialize and tests
  transfer across tasks, roles, and model backbones.

Therefore, memory -> workflow extraction, multi-agent procedural memory, and
small-model reuse cannot be claimed as new in isolation.

## Proposed paper contribution

The research hypothesis is that recurring agent behavior should become an
empirically managed executable policy, not merely more context for the next LLM
call. The proposed contribution is **evidence-constrained continual workflow
synthesis with progressive model shedding**:

1. Discover semantically equivalent trace families behind deterministic
   side-effect and capability gates.
2. Let an LLM propose a restricted typed program, then reject it unless static
   validation and replay over every source episode succeed.
3. Select the cheapest valid executor independently for each computation step.
4. Promote immutable incumbent/challenger versions using held-out executions
   and confidence bounds.
5. Ask the user before reuse and learn from approvals, edits, rejections, and
   full-agent choices.
6. Abstain on uncertain semantic matches, verify state after execution, and
   fall back to the original agent on failure or tool-schema drift.

The key measurable claim is not simply higher task accuracy. It is a Pareto
improvement in verified success, full-reasoning calls, tokens/cost, latency, and
unsafe false matches as experience accumulates.

## Required experiments before a paper claim

### Datasets

1. Mini Office v1 for deterministic debugging and exact ablations.
2. AppWorld using its official environment/evaluator and chronological task
   streams. Run this in a dedicated supported Python environment.
3. OfficeBench/AFTER if their data and licenses are available, because these are
   the closest modern procedural-memory baselines.
4. One framework transfer: learn under LangGraph, replay through an ADK or plain
   Python adapter with equivalent tools.

### Baselines

- full agent with no procedural memory;
- episodic example retrieval;
- Reflexion/ExpeL-style textual insight memory;
- AWM prompt-injected workflows;
- LEGOMem-style orchestrator and worker memories;
- static hand-written workflows as an upper-bound/control;
- this system without confirmation, without verification, without lifecycle,
  and without model shedding.

### Required compiler/retrieval ablation ladder

The committed AppWorld protocols isolate the main value additions:

1. `protocol_exact.json`: exact family + lexical match + deterministic compiler;
2. `protocol_semantic.json`: hybrid family/match + deterministic compiler;
3. `protocol_annotate.json`: hybrid family/match + annotation-only compiler;
4. `protocol.json`: hybrid family/match + validated structural synthesis.

Compare family coverage, offer precision, workflow execution success, official
task success, full-agent avoidance, and total amortized tokens. This determines
whether gains come from semantic discovery, semantic routing, or actual program
synthesis instead of attributing everything to "the LLM."

### Metrics

- official task success and postcondition pass rate;
- workflow offer precision/recall and unsafe false-execution rate;
- full-agent, LLM, and SLM call counts plus model-free task fraction;
- tokens, provider cost, p50/p95 latency, tool calls, and compiler amortization;
- examples required before useful reuse and performance as a function of time;
- quarantine/recovery time under tool schema and environment drift;
- in-domain, paraphrase, compositional, and out-of-distribution transfer;
- calibration of match score and reliability confidence.

### Experimental discipline

Use frozen dataset splits, at least three random seeds, bootstrap confidence
intervals, identical underlying models and tools, and paired significance tests.
Report all compiler calls and failed fallback calls. The current reference token
model must be replaced with actual provider usage for headline cost claims.
For AppWorld, induce on train, calibrate without recompilation on dev, seal the
workflow database, and route only active workflows on test. Official evaluator
feedback may score a completed test attempt but cannot trigger fallback or
update memory. This avoids both cross-task test learning and evaluator-oracle
retry leakage.

AppWorld's three variants per scenario are treated chronologically: the v2
protocol permits synthesis after two verified source variants, while the third
is an unseen shadow execution rather than compiler evidence. Activation still
requires three successful held-out executions and its Wilson lower bound, which
must come from safe reuse across related families rather than relabeling the
two source demonstrations as validation.

## Main threats to validity

- A successful trace may contain unnecessary or accidentally correlated steps.
- Several workflows can share a tool path but have different safety semantics.
- User-provided variables can leak hidden benchmark state if adapters do not
  separate public inputs from evaluator setup.
- Semantic retrieval can produce confident false positives on near-neighbor
  intents; negative examples and abstention must be evaluated, not assumed.
- Workflows can become stale as tool schemas or business policy changes.
- Saving model calls on easy repeated tasks may not improve capability on novel
  tasks; both efficiency and transfer must be reported.

## Current artifact status

Mini Office is an executable engineering benchmark, not a substitute for an
external research benchmark. It proves the plumbing, catches data-flow bugs,
and makes every claimed route inspectable. The AppWorld treatment runner,
bounded control-flow IR, fresh-world fallback, phase guard, freeze manifest,
dependency lock, hybrid discovery/routing, replay-validated LLM synthesis, and
incumbent/challenger lifecycle are implemented. Paid train/dev/test runs with
official evaluation are the next empirical milestone; no AppWorld result is
claimed yet.

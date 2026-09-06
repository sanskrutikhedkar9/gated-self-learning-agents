# AppWorld benchmark protocol

This directory now contains a complete baseline/treatment harness. It is ready
for a small paid smoke run after the repository is committed and preflight is
green. It does **not** imply that SelfLearningFlows has achieved an AppWorld
result yet.

The runner enforces these boundaries:

- train may compile workflows and update reliability;
- dev may update routing reliability but cannot compile from dev outcomes;
- the workflow database is sealed after dev;
- test validates the seal before and after the run and never mutates workflow
  definitions, lifecycle state, or statistics;
- official test evaluation scores the attempt but never selects a fallback;
- an observable workflow execution failure discards the whole AppWorld instance
  and initializes a fresh task world before the full agent runs;
- every model call, including semantic family discovery/routing, variable
  extraction, compilation, and bounded repair, is included in token/call
  accounting by category.
- the fallback code agent receives ranked contracts from AppWorld's real API
  catalogue; invented API names and invisible documentation calls are rejected
  before execution, and a redacted action transcript is retained for diagnosis.

The predeclared settings are in `experiments/appworld/protocol.json`.
It deliberately synthesizes after two verified source episodes. AppWorld has
three variants per scenario, so variant three becomes an unseen shadow trial;
activation still requires three verified shadow executions and the Wilson
confidence gate. Source traces are never counted as held-out successes.

## 1. Environment

AppWorld `0.1.3.post1` needs its own environment. From the SelfLearningFlows
repository in Ubuntu/WSL:

```bash
python3.12 -m venv .venv-appworld
source .venv-appworld/bin/activate
python -m pip install -r experiments/appworld/requirements.txt
python -m pip check
```

`requirements.lock` is the exact dependency set from the verified Linux/Python
3.12 environment. Do not install the LangGraph extra here because that uses a
different Pydantic major version.

Point AppWorld at the checkout that contains `data/` and `experiments/`, then
export the provider key without printing it:

```bash
export APPWORLD_ROOT="$HOME/appworld-v013"
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
echo
```

Never paste a key into a command, config, record, issue, or commit. If a key was
previously exposed, revoke it before running.

## 2. Preflight

The runner refuses a dirty Git repository. Commit the source, then run:

```bash
python -m experiments.appworld.preflight \
  --repository . \
  --protocol experiments/appworld/protocol.json \
  --phase train \
  --split train \
  --dataset "$APPWORLD_ROOT/data/datasets/train.txt"
```

Exit code `0` means no blocker. Exit code `2` means do not spend model budget.
The audit rejects AppWorld's `verification` outputs as training data, rejects
request-only native logs, checks the dependency lock and clean commit, and
validates phase/split/freeze consistency.

Confirm the live AppWorld/schema boundary without a model call:

```bash
python -m experiments.appworld.dryrun --task-id 50e1ac9_1
```

## 3. One-task smoke run

First prove the full-agent boundary and official evaluator with one train task:

```bash
mkdir -p experiments/appworld/outputs

python -m experiments.appworld.treatment_runner \
  --phase train \
  --dataset-name train \
  --condition treatment \
  --experiment-name slf_train_smoke \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_smoke.db \
  --records experiments/appworld/outputs/slf_smoke_records.jsonl \
  --summary experiments/appworld/outputs/slf_smoke_summary.json \
  --compiler-mode structural \
  --family-mode hybrid \
  --matcher-mode semantic \
  --limit 1

head -n 1 "$APPWORLD_ROOT/data/datasets/train.txt" \
  > "$APPWORLD_ROOT/data/datasets/train_smoke.txt"
appworld evaluate --root "$APPWORLD_ROOT" slf_train_smoke train_smoke
```

The first task should route to `full_agent`; no workflow has evidence yet. This
tests the provider, trace recorder, official evaluator, and database boundary,
but it does not yet exercise synthesis. Check the saved summary and AppWorld
score before doing a larger run. Do not reuse the smoke database for the full
train stream because the smoke task ID is already recorded idempotently.

## 4. Treatment learning and dev calibration

Use one database for the treatment sequence:

```bash
python -m experiments.appworld.treatment_runner \
  --phase train \
  --dataset-name train \
  --condition treatment \
  --experiment-name slf_train_v2 \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_v2.db \
  --records experiments/appworld/outputs/slf_v2_records.jsonl \
  --summary experiments/appworld/outputs/slf_train_v2_summary.json \
  --compiler-mode structural \
  --family-mode hybrid \
  --matcher-mode semantic

python -m experiments.appworld.treatment_runner \
  --phase dev \
  --dataset-name dev \
  --condition treatment \
  --experiment-name slf_dev_v2 \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_v2.db \
  --records experiments/appworld/outputs/slf_v2_records.jsonl \
  --summary experiments/appworld/outputs/slf_dev_v2_summary.json \
  --freeze-manifest experiments/appworld/outputs/slf_v2_freeze.json \
  --compiler-mode structural \
  --family-mode hybrid \
  --matcher-mode semantic \
  --seal-after-run

appworld evaluate --root "$APPWORLD_ROOT" slf_dev_v2 dev
self-learning-flows list \
  --database experiments/appworld/outputs/slf_v2.db \
  --scope appworld
```

Do not continue to test unless the list contains at least one `active` workflow.
Candidate means there was insufficient repeated evidence; shadow means there
were not yet enough successful held-out executions. Compilation failures and
their sanitized validator errors remain in workflow metadata for analysis.

Do not choose thresholds after viewing test results. If routing, promotion, or
model settings change after dev, create a new named protocol version and repeat
train/dev.

## 5. Frozen test and matched baseline

Run treatment only after test preflight validates the exact database, protocol,
source commit, and freeze manifest:

```bash
python -m experiments.appworld.preflight \
  --repository . \
  --protocol experiments/appworld/protocol.json \
  --phase test \
  --split test_normal \
  --database experiments/appworld/outputs/slf_v2.db \
  --freeze-manifest experiments/appworld/outputs/slf_v2_freeze.json

python -m experiments.appworld.treatment_runner \
  --phase test \
  --dataset-name test_normal \
  --condition treatment \
  --experiment-name slf_test_normal_v2 \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_v2.db \
  --summary experiments/appworld/outputs/slf_test_normal_v2_summary.json \
  --freeze-manifest experiments/appworld/outputs/slf_v2_freeze.json \
  --compiler-mode structural \
  --family-mode hybrid \
  --matcher-mode semantic

python -m experiments.appworld.treatment_runner \
  --phase test \
  --dataset-name test_normal \
  --condition baseline \
  --experiment-name slf_baseline_test_normal_v2 \
  --model gpt-4o \
  --database experiments/appworld/outputs/baseline_v2.db \
  --summary experiments/appworld/outputs/slf_baseline_test_normal_v2_summary.json

appworld evaluate --root "$APPWORLD_ROOT" slf_test_normal_v2 test_normal
appworld evaluate --root "$APPWORLD_ROOT" slf_baseline_test_normal_v2 test_normal
```

Use the same model, random seed, maximum interactions, code commit, and task
order for both conditions. Repeat with predeclared seeds for confidence
intervals; do not report one lucky run.

## 6. Ablations after the first full treatment

Do not pay for all ablations until the full treatment completes without runner
errors. The committed protocols isolate the contribution:

| Protocol | Compiler | Family discovery | Request matching |
|---|---|---|---|
| `protocol_exact.json` | deterministic | exact | lexical |
| `protocol_semantic.json` | deterministic | hybrid | semantic |
| `protocol_annotate.json` | annotation only | hybrid | semantic |
| `protocol.json` | replay-validated structural LLM | hybrid | semantic |

Each ablation needs its own fresh database, records, experiment names, and freeze
manifest. Pass the corresponding `--protocol`, `--compiler-mode`,
`--family-mode`, and `--matcher-mode`; the runner aborts if CLI settings differ
from the predeclared protocol. The matched full-agent baseline can be reused for
paired comparisons when model, prompt, task order, seed, and source commit are
identical.

## What is recorded

`treatment_runner.py` writes aggregate per-task routing and cost metrics to the
SQLite `metrics` table and, with `--summary`, to a standalone JSON report.
Provider usage is split into agent, workflow compute, and structured overhead;
the latter is further categorized as family discovery, workflow synthesis,
semantic routing, and variable extraction. Optional JSONL traces contain
arguments, result, success, latency, evaluator provenance, used tool schemas,
and schema hashes. Credential-shaped fields are replaced with stable
within-episode markers before storage. The compiler receives symbolic shapes and
references, not concrete stored results or argument values.

Raw AppWorld data, output databases, task traces, and credentials stay ignored
by Git. Commit only aggregate, non-sensitive results allowed by AppWorld's data
policy.

The bundled code agent is deliberately simple and is used unchanged in baseline
and fallback routes. A future adapter can replace it with another agent, but a
paper comparison must keep the underlying agent identical across conditions.

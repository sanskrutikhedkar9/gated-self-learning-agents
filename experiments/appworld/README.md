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
- every model call, including variable extraction and model-assisted compilation,
  is included in token/call accounting.

The predeclared settings are in `experiments/appworld/protocol.json`.

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
python experiments/appworld/preflight.py \
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
python experiments/appworld/dryrun.py --task-id 50e1ac9_1
```

## 3. One-task smoke run

First prove the full-agent boundary and official evaluator with one train task:

```bash
mkdir -p experiments/appworld/outputs

python experiments/appworld/treatment_runner.py \
  --phase train \
  --dataset-name train \
  --condition treatment \
  --experiment-name slf_train_smoke \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_smoke.db \
  --records experiments/appworld/outputs/slf_smoke_records.jsonl \
  --limit 1

appworld evaluate slf_train_smoke train
```

The first task should route to `full_agent`; no workflow has evidence yet. Check
the printed provider token counts and the official AppWorld score before doing a
larger run.

## 4. Treatment learning and dev calibration

Use one database for the treatment sequence:

```bash
python experiments/appworld/treatment_runner.py \
  --phase train \
  --dataset-name train \
  --condition treatment \
  --experiment-name slf_train_v1 \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_v1.db \
  --records experiments/appworld/outputs/slf_v1_records.jsonl

python experiments/appworld/treatment_runner.py \
  --phase dev \
  --dataset-name dev \
  --condition treatment \
  --experiment-name slf_dev_v1 \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_v1.db \
  --records experiments/appworld/outputs/slf_v1_records.jsonl \
  --freeze-manifest experiments/appworld/outputs/slf_v1_freeze.json \
  --seal-after-run

appworld evaluate slf_dev_v1 dev
```

Do not choose thresholds after viewing test results. If routing, promotion, or
model settings change after dev, create a new named protocol version and repeat
train/dev.

## 5. Frozen test and matched baseline

Run treatment only after test preflight validates the exact database, protocol,
source commit, and freeze manifest:

```bash
python experiments/appworld/preflight.py \
  --repository . \
  --protocol experiments/appworld/protocol.json \
  --phase test \
  --split test_normal \
  --database experiments/appworld/outputs/slf_v1.db \
  --freeze-manifest experiments/appworld/outputs/slf_v1_freeze.json

python experiments/appworld/treatment_runner.py \
  --phase test \
  --dataset-name test_normal \
  --condition treatment \
  --experiment-name slf_test_normal_v1 \
  --model gpt-4o \
  --database experiments/appworld/outputs/slf_v1.db \
  --freeze-manifest experiments/appworld/outputs/slf_v1_freeze.json

python experiments/appworld/treatment_runner.py \
  --phase test \
  --dataset-name test_normal \
  --condition baseline \
  --experiment-name slf_baseline_test_normal_v1 \
  --model gpt-4o \
  --database experiments/appworld/outputs/baseline_v1.db

appworld evaluate slf_test_normal_v1 test_normal
appworld evaluate slf_baseline_test_normal_v1 test_normal
```

Use the same model, random seed, maximum interactions, code commit, and task
order for both conditions. Repeat with predeclared seeds for confidence
intervals; do not report one lucky run.

## What is recorded

`treatment_runner.py` writes aggregate per-task routing and cost metrics to the
SQLite `metrics` table. Optional JSONL traces are rich enough to compile data
flow: each call contains arguments, result, success, latency, evaluator
provenance, and schema hashes. Credential-shaped fields are replaced with stable
within-episode markers before storage.

Raw AppWorld data, output databases, task traces, and credentials stay ignored
by Git. Commit only aggregate, non-sensitive results allowed by AppWorld's data
policy.

The bundled code agent is deliberately simple and is used unchanged in baseline
and fallback routes. A future adapter can replace it with another agent, but a
paper comparison must keep the underlying agent identical across conditions.

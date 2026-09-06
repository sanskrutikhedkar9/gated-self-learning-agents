"""Ingest normalized AppWorld agent records into procedural memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_learning_flows import SelfLearningFlowEngine, SQLiteStore
from self_learning_flows.adapters.appworld import AppWorldTraceAdapter


def ingest(path: Path, database: Path) -> dict[str, int]:
    store = SQLiteStore(database)
    engine = SelfLearningFlowEngine(store)
    adapter = AppWorldTraceAdapter()
    records = verified = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                episode = adapter.normalize(**record)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid record at line {line_number}: {exc}") from exc
            engine.observe(episode)
            records += 1
            verified += int(episode.verified and episode.success)
    statuses: dict[str, int] = {}
    for workflow in store.list_workflows(scope="appworld"):
        key = str(workflow.status)
        statuses[key] = statuses.get(key, 0) + 1
    return {"records": records, "verified_successes": verified, **statuses}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--database", type=Path, default=Path("appworld_workflows.db"))
    args = parser.parse_args()
    print(json.dumps(ingest(args.records, args.database), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

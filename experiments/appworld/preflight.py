"""Fail-fast audit for AppWorld data, traces, and online reuse opportunity."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from self_learning_flows.adapters.appworld import AppWorldTraceAdapter
from self_learning_flows.discovery import structural_signature
from self_learning_flows.research_protocol import (
    ExperimentPhase,
    ProtocolConfig,
    ProtocolGuard,
    ProtocolViolation,
)
from self_learning_flows.storage import SQLiteStore

_VARIANT_SUFFIX = re.compile(r"_\d+$")
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:access_token|refresh_token|api_key|password|authorization|secret|token)(?:$|_)",
    re.IGNORECASE,
)


def audit_dataset(path: Path, *, min_observations: int) -> dict[str, Any]:
    task_ids = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    scenario_counts = Counter(_VARIANT_SUFFIX.sub("", task_id) for task_id in task_ids)
    same_scenario_routes = sum(
        max(0, count - min_observations) for count in scenario_counts.values()
    )
    return {
        "path": str(path),
        "tasks": len(task_ids),
        "scenarios": len(scenario_counts),
        "variants_per_scenario": {
            "minimum": min(scenario_counts.values(), default=0),
            "maximum": max(scenario_counts.values(), default=0),
        },
        "min_observations_before_offer": min_observations,
        "maximum_same_scenario_online_routes": same_scenario_routes,
    }


def _has_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _SENSITIVE_KEY.search(str(key)) or _has_sensitive_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_sensitive_key(item) for item in value)
    return False


def audit_native_outputs(path: Path, *, min_observations: int) -> dict[str, Any]:
    task_root = path / "tasks"
    task_directories = sorted(item for item in task_root.glob("*") if item.is_dir())
    exact_paths: Counter[tuple[str, ...]] = Counter()
    calls = incomplete_calls = sensitive_calls = logs = 0
    parse_errors: list[str] = []
    for task_directory in task_directories:
        log_path = task_directory / "logs" / "api_calls.jsonl"
        if not log_path.exists():
            continue
        logs += 1
        urls: list[str] = []
        for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                call = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(f"{task_directory.name}:{line_number}: {exc.msg}")
                continue
            calls += 1
            urls.append(str(call.get("url", "")))
            if "result" not in call or type(call.get("success")) is not bool:
                incomplete_calls += 1
            if _has_sensitive_key(call.get("data", {})):
                sensitive_calls += 1
        exact_paths[tuple(urls)] += 1
    theoretical_routes = sum(max(0, count - min_observations) for count in exact_paths.values())
    return {
        "path": str(path),
        "is_appworld_ground_truth_verification": path.name.lower() == "verification",
        "task_directories": len(task_directories),
        "api_call_logs": logs,
        "calls": calls,
        "calls_missing_result_or_success": incomplete_calls,
        "calls_with_sensitive_fields": sensitive_calls,
        "unique_exact_call_paths": len(exact_paths),
        "theoretical_exact_path_routes_after_threshold": theoretical_routes,
        "parse_errors": parse_errors[:10],
        "ingestible": bool(logs) and not incomplete_calls and not parse_errors,
    }


def audit_records(path: Path, *, min_observations: int) -> dict[str, Any]:
    adapter = AppWorldTraceAdapter()
    seen: Counter[str] = Counter()
    records = valid = verified_successes = potential_routes = 0
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        records += 1
        try:
            episode = adapter.normalize(**json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        valid += 1
        verified_successes += int(episode.success and episode.verified)
        signature = structural_signature(episode)
        if seen[signature] >= min_observations:
            potential_routes += 1
        seen[signature] += 1
    return {
        "path": str(path),
        "records": records,
        "valid_records": valid,
        "verified_successes": verified_successes,
        "unique_structural_signatures": len(seen),
        "potential_online_routes_after_threshold": potential_routes,
        "errors": errors[:10],
        "ingestible": records > 0 and valid == records,
    }


def audit_repository(path: Path) -> dict[str, Any]:
    repository = path.resolve()

    def git(*arguments: str) -> str:
        command = ["git", "-c", f"safe.directory={repository.as_posix()}", *arguments]
        return subprocess.check_output(command, cwd=repository, text=True).rstrip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=all").splitlines()
        tracked = set(git("ls-files").splitlines())
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"path": str(repository), "valid_git_repository": False, "error": str(exc)}
    lock_candidates = (
        repository / "uv.lock",
        repository / "experiments" / "appworld" / "requirements.lock",
    )
    return {
        "path": str(repository),
        "valid_git_repository": True,
        "commit": commit,
        "clean": not status,
        "dirty_paths": [line[3:] for line in status[:30]],
        "dependency_locks": [
            str(item.relative_to(repository)) for item in lock_candidates if item.exists()
        ],
        "tracked_secret_files": sorted(
            name for name in tracked if name in {".env", ".env.local", ".env.production"}
        ),
    }


def audit_protocol(
    config_path: Path,
    *,
    phase: str,
    split: str,
    database: Path | None = None,
    freeze_manifest: Path | None = None,
) -> dict[str, Any]:
    try:
        config = ProtocolConfig.load(config_path)
        guard = ProtocolGuard(config, ExperimentPhase(phase), split)
        result: dict[str, Any] = {
            "path": str(config_path),
            "name": config.name,
            "phase": phase,
            "split": split,
            "allows_learning": guard.allows_learning,
            "allows_calibration": guard.allows_calibration,
            "fresh_world_fallback": config.fresh_world_fallback,
            "valid": True,
        }
        if phase == ExperimentPhase.TEST.value:
            if database is None or freeze_manifest is None:
                raise ProtocolViolation("test preflight requires --database and --freeze-manifest")
            if not database.exists() or not freeze_manifest.exists():
                raise ProtocolViolation("test database or freeze manifest does not exist")
            store = SQLiteStore(database)
            guard.validate_frozen(store, freeze_manifest)
            active_workflows = store.list_workflows(statuses={"active"})
            if not active_workflows:
                raise ProtocolViolation("frozen test database contains no active workflows")
            result["active_workflows"] = len(active_workflows)
            result["freeze_valid"] = True
        return result
    except (OSError, ValueError, ProtocolViolation, json.JSONDecodeError) as exc:
        return {
            "path": str(config_path),
            "phase": phase,
            "split": split,
            "valid": False,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--native-outputs", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--phase", choices=[item.value for item in ExperimentPhase])
    parser.add_argument("--split")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--min-observations", type=int, default=3)
    args = parser.parse_args()
    if not any((args.dataset, args.native_outputs, args.records, args.repository, args.protocol)):
        parser.error(
            "pass at least one audit input: --dataset, --native-outputs, --records, "
            "--repository, or --protocol"
        )
    if args.min_observations < 1:
        parser.error("--min-observations must be positive")

    report: dict[str, Any] = {"blockers": [], "warnings": []}
    if args.dataset:
        dataset = audit_dataset(args.dataset, min_observations=args.min_observations)
        report["dataset"] = dataset
        if dataset["maximum_same_scenario_online_routes"] == 0:
            report["warnings"].append(
                "The promotion threshold leaves no later variant in the same scenario for reuse."
            )
    if args.native_outputs:
        native = audit_native_outputs(args.native_outputs, min_observations=args.min_observations)
        report["native_outputs"] = native
        if native["is_appworld_ground_truth_verification"]:
            report["blockers"].append(
                "The 'verification' experiment is produced by AppWorld's ground-truth "
                "task validator; it is not an agent baseline and must not train the system."
            )
        if not native["ingestible"]:
            report["blockers"].append(
                "Native AppWorld request logs lack call results/success and cannot "
                "compile data flow."
            )
        if native["calls_with_sensitive_fields"]:
            report["warnings"].append(
                "Native logs contain sensitive fields; keep outputs ignored and use "
                "redacted records."
            )
    if args.records:
        records = audit_records(args.records, min_observations=args.min_observations)
        report["records"] = records
        if not records["ingestible"]:
            report["blockers"].append(
                "Normalized records failed strict provenance/trace validation."
            )
        elif records["potential_online_routes_after_threshold"] == 0:
            report["warnings"].append(
                "No record arrives after its structural signature reaches the promotion threshold."
            )
    if args.repository:
        repository = audit_repository(args.repository)
        report["repository"] = repository
        if not repository.get("valid_git_repository"):
            report["blockers"].append("Source directory is not a valid Git repository.")
        else:
            if not repository["clean"]:
                report["blockers"].append(
                    "Repository is dirty; experiment source provenance is not reproducible."
                )
            if not repository["dependency_locks"]:
                report["blockers"].append("No dependency lock is present.")
            if repository["tracked_secret_files"]:
                report["blockers"].append("A secret-bearing .env file is tracked by Git.")
    if args.protocol:
        if not args.phase or not args.split:
            parser.error("--protocol requires --phase and --split")
        protocol = audit_protocol(
            args.protocol,
            phase=args.phase,
            split=args.split,
            database=args.database,
            freeze_manifest=args.freeze_manifest,
        )
        report["protocol"] = protocol
        if not protocol["valid"]:
            report["blockers"].append("The declared phase/split/freeze protocol is invalid.")
        elif not protocol["fresh_world_fallback"]:
            report["blockers"].append("Protocol does not require fresh-world fallback.")
    print(json.dumps(report, indent=2))
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from Detection_Agent import run_detection
from Evaluation_Agent import run_evaluation
from Repair_Agent import run_repair
from progress_tracker import ProgressTracker


PHASES = ("pain", "random", "test", "full")
PHASE_DATASET_DEFAULTS = {
    "pain": Path("data") / "Dataset_pain300_fixed.xlsx",
    "random": Path("data") / "Dataset_random300_fixed.xlsx",
    "test": Path("data") / "Dataset_test500_fixed.xlsx",
}



def resolve_full_dataset_default() -> Path:
    p1 = Path("Data") / "Dataset.xlsx"
    p2 = Path("data") / "Dataset.xlsx"
    if p1.exists():
        return p1
    return p2


def resolve_dataset_path(path: Path | None, phase: str) -> Path:
    if path is not None:
        return path
    if phase in PHASE_DATASET_DEFAULTS:
        return PHASE_DATASET_DEFAULTS[phase]
    return resolve_full_dataset_default()


def allocate_run_output_dir(base_output_dir: Path) -> tuple[Path, str]:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"run_(\d{3})$")
    max_idx = 0
    for p in base_output_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.fullmatch(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    run_idx = max_idx + 1
    run_id = f"run_{run_idx:03d}"
    run_output_dir = base_output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir, run_id


def resolve_experiment_dir(base_output_dir: Path, experiment_id: str | None) -> tuple[Path, str]:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if experiment_id:
        run_output_dir = base_output_dir / experiment_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        return run_output_dir, experiment_id
    return allocate_run_output_dir(base_output_dir)


def ensure_phase_roots(experiment_dir: Path) -> None:
    for phase in ("pain", "random", "test"):
        (experiment_dir / phase).mkdir(parents=True, exist_ok=True)


def allocate_attempt_dir(phase_root: Path, explicit_attempt: int | None) -> tuple[Path, str]:
    if explicit_attempt is not None:
        attempt_name = f"attempt_{explicit_attempt:03d}"
        attempt_dir = phase_root / attempt_name
        if attempt_dir.exists():
            status_path = attempt_dir / "pipeline_status.json"
            if not status_path.exists():
                raise FileExistsError(f"Attempt directory already exists: {attempt_dir}")
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
            if status != "failed":
                raise FileExistsError(f"Attempt directory already exists: {attempt_dir}")
            return attempt_dir, attempt_name
        attempt_dir.mkdir(parents=True, exist_ok=False)
        return attempt_dir, attempt_name

    pattern = re.compile(r"attempt_(\d{3})$")
    max_idx = 0
    for p in phase_root.iterdir():
        if not p.is_dir():
            continue
        m = pattern.fullmatch(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    attempt_idx = max_idx + 1
    attempt_name = f"attempt_{attempt_idx:03d}"
    attempt_dir = phase_root / attempt_name
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir, attempt_name


def update_experiment_manifest(
    experiment_dir: Path,
    *,
    phase: str,
    attempt: str,
    dataset: Path,
    status: str,
    output_dir: Path,
    error: str = "",
) -> None:
    manifest_path = experiment_dir / "experiment_status.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {
            "experiment_id": experiment_dir.name,
            "phases": {"pain": [], "random": [], "test": [], "full": []},
        }

    phases = data.setdefault("phases", {})
    history = phases.setdefault(phase, [])
    if status == "running":
        history.append(
            {
                "attempt": attempt,
                "status": "running",
                "dataset": str(dataset),
                "output_dir": str(output_dir),
            }
        )
    else:
        for item in reversed(history):
            if item.get("attempt") == attempt and item.get("status") == "running":
                item["status"] = status
                if error:
                    item["error"] = error
                break
        else:
            history.append(
                {
                    "attempt": attempt,
                    "status": status,
                    "dataset": str(dataset),
                    "output_dir": str(output_dir),
                    **({"error": error} if error else {}),
                }
            )
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Detection -> Repair -> Evaluation pipeline")
    p.add_argument("--phase", choices=PHASES, default="full", help="Experiment phase. pain/random/test/full")
    p.add_argument("--experiment-id", type=str, default=None, help="Reuse an existing run folder (e.g., run_011)")
    p.add_argument("--attempt", type=int, default=None, help="Explicit attempt index under phase folder (2 -> attempt_002)")
    p.add_argument("--dataset", type=Path, default=None, help="Override dataset path. Default depends on --phase.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--confidence-threshold", type=float, default=0.75)
    p.add_argument("--second-check-threshold", type=float, default=0.80)
    p.add_argument("--borderline-threshold", type=float, default=0.90)
    p.add_argument("--uncertainty-margin-threshold", type=float, default=0.18)
    p.add_argument("--detection-temperature", type=float, default=0.0)
    p.add_argument("--repair-quality-threshold", type=float, default=0.70)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phase = args.phase
    dataset = resolve_dataset_path(args.dataset, phase)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    experiment_dir, run_id = resolve_experiment_dir(args.output_dir, args.experiment_id)
    ensure_phase_roots(experiment_dir)
    phase_root = experiment_dir / phase
    if phase_root.name == "full":
        phase_root.mkdir(parents=True, exist_ok=True)
    run_output_dir, attempt_name = allocate_attempt_dir(phase_root, args.attempt)
    update_experiment_manifest(
        experiment_dir,
        phase=phase,
        attempt=attempt_name,
        dataset=dataset,
        status="running",
        output_dir=run_output_dir,
    )
    tracker = ProgressTracker(run_output_dir)

    tracker.update(
        stage="pipeline",
        status="running",
        current=0,
        total=3,
        message="Pipeline started",
        extra={
            "dataset": str(dataset),
            "run_id": run_id,
            "phase": phase,
            "attempt": attempt_name,
            "run_output_dir": str(run_output_dir),
        },
    )

    try:
        tracker.update(stage="pipeline", status="running", current=1, total=3, message="Running detection")
        det_paths = run_detection(
            dataset_path=dataset,
            output_dir=run_output_dir,
            limit=args.limit,
            confidence_threshold=args.confidence_threshold,
            second_check_threshold=args.second_check_threshold,
            borderline_threshold=args.borderline_threshold,
            uncertainty_margin_threshold=args.uncertainty_margin_threshold,
            temperature=args.detection_temperature,
            fail_fast=True,
            progress_callback=tracker.callback,
        )

        tracker.update(stage="pipeline", status="running", current=2, total=3, message="Running repair")
        rep_paths = run_repair(
            dataset_path=dataset,
            detection_trace_path=det_paths["detection_trace_jsonl"],
            output_dir=run_output_dir,
            limit=args.limit,
            fail_fast=True,
            progress_callback=tracker.callback,
        )

        tracker.update(stage="pipeline", status="running", current=3, total=3, message="Running evaluation")
        eval_paths = run_evaluation(
            dataset_path=dataset,
            detection_trace_path=det_paths["detection_trace_jsonl"],
            repair_trace_path=rep_paths["repair_trace_jsonl"],
            output_dir=run_output_dir,
            limit=args.limit,
            confidence_threshold=args.confidence_threshold,
            repair_quality_threshold=args.repair_quality_threshold,
            progress_callback=tracker.callback,
        )
    except Exception as exc:
        tracker.update(
            stage="pipeline",
            status="failed",
            current=None,
            total=3,
            message="Pipeline failed",
            extra={"error": str(exc)},
        )
        update_experiment_manifest(
            experiment_dir,
            phase=phase,
            attempt=attempt_name,
            dataset=dataset,
            status="failed",
            output_dir=run_output_dir,
            error=str(exc),
        )
        raise

    tracker.update(
        stage="pipeline",
        status="completed",
        current=3,
        total=3,
        message="Pipeline finished",
        extra={
            "run_id": run_id,
            "run_output_dir": str(run_output_dir),
            "detection_results": str(det_paths["detection_results_jsonl"]),
            "repair_results": str(rep_paths["repair_results_jsonl"]),
            "evaluation_report": str(eval_paths["evaluation_report_json"]),
        },
    )
    update_experiment_manifest(
        experiment_dir,
        phase=phase,
        attempt=attempt_name,
        dataset=dataset,
        status="completed",
        output_dir=run_output_dir,
    )

    print("Pipeline finished.")
    print(f"Run id: {run_id}")
    print(f"Phase: {phase}")
    print(f"Attempt: {attempt_name}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Run output dir: {run_output_dir}")
    print(f"Detection results: {det_paths['detection_results_jsonl']}")
    print(f"Repair results: {rep_paths['repair_results_jsonl']}")
    print(f"Evaluation report: {eval_paths['evaluation_report_json']}")


if __name__ == "__main__":
    main()

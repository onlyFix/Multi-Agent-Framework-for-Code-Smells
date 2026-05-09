from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ProgressTracker:
    def __init__(self, output_dir: Path, print_to_console: bool = True) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "pipeline_status.json"
        self.events_path = self.output_dir / "pipeline_events.jsonl"
        self.print_to_console = print_to_console

    def update(
        self,
        *,
        stage: str,
        status: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: str = "",
        sample_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "stage": stage,
            "status": status,
            "message": message,
        }

        if current is not None:
            payload["current"] = current
        if total is not None:
            payload["total"] = total
            if total > 0 and current is not None:
                payload["progress"] = round((current / total) * 100.0, 2)
        if sample_id is not None:
            payload["sample_id"] = sample_id
        if extra:
            payload["extra"] = extra

        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if self.print_to_console:
            print(self._format_console_line(payload), flush=True)

    def _format_console_line(self, payload: Dict[str, Any]) -> str:
        stage = payload.get("stage", "")
        status = payload.get("status", "")
        message = payload.get("message", "")
        sample_id = payload.get("sample_id")
        current = payload.get("current")
        total = payload.get("total")
        progress = payload.get("progress")
        extra = payload.get("extra", {})

        parts = [f"[{stage}:{status}]"]
        if current is not None and total is not None:
            if progress is not None:
                parts.append(f"{current}/{total} ({progress}%)")
            else:
                parts.append(f"{current}/{total}")
        if sample_id is not None:
            parts.append(f"sample_id={sample_id}")
        if message:
            parts.append(message)
        if isinstance(extra, dict) and extra:
            kv = ", ".join(f"{k}={v}" for k, v in extra.items())
            parts.append(f"extra: {kv}")
        return " ".join(parts)

    def callback(
        self,
        stage: str,
        status: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: str = "",
        sample_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.update(
            stage=stage,
            status=status,
            current=current,
            total=total,
            message=message,
            sample_id=sample_id,
            extra=extra,
        )

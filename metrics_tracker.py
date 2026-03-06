from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class MetricRecord:
    paper_name: str
    run_at_utc: str
    score: float
    note: str
    result_path: str
    log_path: str


class MetricsTracker:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def ensure_file(self) -> None:
        if self.csv_path.exists():
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["paper_name", "run_at_utc", "score", "note", "result_path", "log_path"])

    def append(self, paper_name: str, score: float, note: str, result_path: Path, log_path: Path) -> MetricRecord:
        self.ensure_file()
        run_at_utc = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        record = MetricRecord(
            paper_name=paper_name,
            run_at_utc=run_at_utc,
            score=score,
            note=note,
            result_path=str(result_path),
            log_path=str(log_path),
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    record.paper_name,
                    record.run_at_utc,
                    f"{record.score:.6f}",
                    record.note,
                    record.result_path,
                    record.log_path,
                ]
            )
        return record

    def recent_summary(self, limit: int = 10) -> str:
        self.ensure_file()
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return "최근 실험 데이터가 없습니다."
        rows = rows[-limit:]
        lines = ["최근 실험 통계:"]
        for row in rows:
            lines.append(
                f"- {row['run_at_utc']} | {row['paper_name']} | score={row['score']} | note={row['note']}"
            )
        return "\n".join(lines)

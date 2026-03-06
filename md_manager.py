from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


CHECK_PATTERN = re.compile(r"^- \[(?P<mark>[ xX])\] (?P<title>.+)$")
REQUEST_PATTERN = re.compile(
    r"^- \[ \] request_id: (?P<id>[^|]+) \| status: (?P<status>[^|]+) \| title: (?P<title>[^|]+) "
    r"\| created_at_utc: (?P<created>[^|]+) \| started_at_utc: (?P<started>[^|]+) "
    r"\| completed_at_utc: (?P<completed>[^|]+) \| updated_at_utc: (?P<updated>[^|]+) "
    r"\| files: (?P<files>[^|]+) \| summary: (?P<summary>.+)$"
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkRequest:
    request_id: str
    status: str
    title: str
    created_at_utc: str
    started_at_utc: str
    completed_at_utc: str
    updated_at_utc: str
    files: str
    summary: str


class ChecklistManager:
    def __init__(self, paper_list_path: Path) -> None:
        self.paper_list_path = paper_list_path

    def ensure_default(self) -> None:
        if self.paper_list_path.exists():
            return
        default = (
            "# AI 논문 체크리스트\n\n"
            "## LLM\n"
            "- [ ] Attention Is All You Need\n\n"
            "## Diffusion\n"
            "- [ ] DDPM\n\n"
            "## Vision\n"
            "- [ ] Vision Transformer\n"
        )
        self.paper_list_path.write_text(default, encoding="utf-8")

    def add_paper(self, category: str, title: str) -> bool:
        self.ensure_default()
        content = self.paper_list_path.read_text(encoding="utf-8")
        if f"- [ ] {title}" in content or f"- [x] {title}" in content:
            return False

        if f"## {category}" not in content:
            content = f"{content.rstrip()}\n\n## {category}\n"
        lines = content.splitlines()
        insert_idx = len(lines)
        for idx, line in enumerate(lines):
            if line.strip() == f"## {category}":
                insert_idx = idx + 1
                while insert_idx < len(lines) and not lines[insert_idx].startswith("## "):
                    insert_idx += 1
                break
        lines.insert(insert_idx, f"- [ ] {title}")
        self.paper_list_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return True

    def check_paper(self, title: str) -> bool:
        self.ensure_default()
        lines = self.paper_list_path.read_text(encoding="utf-8").splitlines()
        changed = False
        for i, line in enumerate(lines):
            if line.strip() == f"- [ ] {title}":
                lines[i] = f"- [x] {title}"
                changed = True
                break
        if changed:
            self.paper_list_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return changed

    def remove_paper(self, title: str) -> bool:
        self.ensure_default()
        lines = self.paper_list_path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        removed = False
        for line in lines:
            if line.strip() in {f"- [ ] {title}", f"- [x] {title}"}:
                removed = True
                continue
            new_lines.append(line)
        if removed:
            self.paper_list_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
        return removed

    def render(self) -> str:
        self.ensure_default()
        return self.paper_list_path.read_text(encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_request_id() -> str:
    dt = datetime.now(tz=timezone.utc)
    return f"req-{dt.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class WorkBoardManager:
    def __init__(self, path: Path) -> None:
        self.path = path

    def ensure_exists(self) -> None:
        if self.path.exists():
            self._repair_if_needed()
            return
        now = utc_now_iso()
        text = (
            "# 작업 보드\n\n"
            "## 실행 규칙\n"
            "- 프로그램 시작 시 이 파일을 먼저 로드\n"
            "- 사용자 요청이 들어오면 요청 목록 자동 업데이트\n"
            "- 상태(todo/doing/done)와 수정 파일, 완료시각(UTC) 자동 반영\n\n"
            "## 보드 요약\n"
            "- total: 0\n"
            "- todo: 0\n"
            "- doing: 0\n"
            "- done: 0\n"
            f"- last_updated_utc: {now}\n\n"
            "## 요청 목록\n\n"
            "## 작업 로그\n\n"
            "## 무결성 규칙\n"
            "- 요청 목록의 각 request_id는 유일해야 한다.\n"
            "- done 상태는 completed_at_utc가 반드시 있어야 한다.\n"
            "- doing 상태는 started_at_utc가 반드시 있어야 한다.\n"
            "- 작업 로그의 request_id는 요청 목록에 존재해야 한다.\n"
            "- 보드 요약 카운트는 요청 목록 상태 집계와 일치해야 한다.\n"
        )
        self._atomic_write(text)

    def _repair_if_needed(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        required_sections = [
            "## 실행 규칙",
            "## 보드 요약",
            "## 요청 목록",
            "## 작업 로그",
            "## 무결성 규칙",
        ]
        changed = False
        for section in required_sections:
            if section not in text:
                text += f"\n{section}\n"
                changed = True
        if changed:
            self._atomic_write(text.rstrip() + "\n")

    def _atomic_write(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self.path.parent) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)

    def _split_sections(self, text: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current = "__top__"
        sections[current] = []
        for line in text.splitlines():
            if line.startswith("## "):
                current = line.strip()
                sections.setdefault(current, [])
            else:
                sections.setdefault(current, []).append(line)
        return sections

    def _parse_requests(self, lines: list[str]) -> list[WorkRequest]:
        items: list[WorkRequest] = []
        for line in lines:
            m = REQUEST_PATTERN.match(line.strip())
            if not m:
                continue
            items.append(
                WorkRequest(
                    request_id=m.group("id").strip(),
                    status=m.group("status").strip(),
                    title=m.group("title").strip(),
                    created_at_utc=m.group("created").strip(),
                    started_at_utc=m.group("started").strip(),
                    completed_at_utc=m.group("completed").strip(),
                    updated_at_utc=m.group("updated").strip(),
                    files=m.group("files").strip(),
                    summary=m.group("summary").strip(),
                )
            )
        return items

    def _validate_transition(self, old: str, new: str) -> bool:
        allowed = {"todo": ["doing"], "doing": ["done"], "done": []}
        return new in allowed.get(old, [])

    def _format_request(self, req: WorkRequest) -> str:
        return (
            "- [ ] request_id: "
            f"{req.request_id} | status: {req.status} | title: {req.title} | created_at_utc: {req.created_at_utc} "
            f"| started_at_utc: {req.started_at_utc} | completed_at_utc: {req.completed_at_utc} "
            f"| updated_at_utc: {req.updated_at_utc} | files: {req.files} | summary: {req.summary}"
        )

    def _rebuild(self, sections: dict[str, list[str]], requests: list[WorkRequest], append_log: str | None) -> str:
        summary = {
            "total": len(requests),
            "todo": len([r for r in requests if r.status == "todo"]),
            "doing": len([r for r in requests if r.status == "doing"]),
            "done": len([r for r in requests if r.status == "done"]),
            "last_updated_utc": utc_now_iso(),
        }
        sections["## 보드 요약"] = [
            f"- total: {summary['total']}",
            f"- todo: {summary['todo']}",
            f"- doing: {summary['doing']}",
            f"- done: {summary['done']}",
            f"- last_updated_utc: {summary['last_updated_utc']}",
            "",
        ]
        sections["## 요청 목록"] = [self._format_request(r) for r in requests] + [""]
        logs = [line for line in sections.get("## 작업 로그", []) if line.strip()]
        if append_log:
            logs.append(append_log)
        sections["## 작업 로그"] = logs + [""]

        order = [
            "__top__",
            "## 실행 규칙",
            "## 보드 요약",
            "## 요청 목록",
            "## 작업 로그",
            "## 무결성 규칙",
        ]
        chunks: list[str] = []
        for key in order:
            if key == "__top__":
                if sections.get(key):
                    chunks.extend(sections[key])
                continue
            chunks.append(key)
            chunks.extend(sections.get(key, [""]))
            if chunks and chunks[-1] != "":
                chunks.append("")
        return "\n".join(chunks).rstrip() + "\n"

    def _save_and_verify(self, content: str) -> None:
        self._atomic_write(content)
        try:
            reloaded = self.path.read_text(encoding="utf-8")
            self._split_sections(reloaded)
        except Exception as exc:
            logger.exception("work_board.md 저장 검증 실패: %s", exc)

    def create_request(self, title: str) -> str:
        self.ensure_exists()
        text = self.path.read_text(encoding="utf-8")
        sections = self._split_sections(text)
        requests = self._parse_requests(sections.get("## 요청 목록", []))

        req_id = make_request_id()
        while any(r.request_id == req_id for r in requests):
            req_id = make_request_id()

        now = utc_now_iso()
        new_req = WorkRequest(
            request_id=req_id,
            status="todo",
            title=title,
            created_at_utc=now,
            started_at_utc="-",
            completed_at_utc="-",
            updated_at_utc=now,
            files="-",
            summary="-",
        )
        requests.append(new_req)
        log_line = (
            f"- {now} | request_id: {req_id} | event: created | status: todo | title: {title} "
            "| files: - | note: 요청 등록"
        )
        self._save_and_verify(self._rebuild(sections, requests, log_line))
        return req_id

    def update_status(self, request_id: str, new_status: str, files: list[str] | None = None, summary: str = "-") -> None:
        self.ensure_exists()
        files = sorted(set(files or []))
        text = self.path.read_text(encoding="utf-8")
        sections = self._split_sections(text)
        requests = self._parse_requests(sections.get("## 요청 목록", []))

        target = next((r for r in requests if r.request_id == request_id), None)
        if target is None:
            logger.warning("요청 ID를 찾지 못함: %s", request_id)
            return
        old_status = target.status
        if old_status != new_status:
            if not self._validate_transition(old_status, new_status):
                logger.warning("잘못된 상태 전이: %s -> %s", old_status, new_status)
                return
        now = utc_now_iso()
        target.status = new_status
        target.updated_at_utc = now
        if new_status == "doing" and target.started_at_utc == "-":
            target.started_at_utc = now
        if new_status == "done":
            target.completed_at_utc = now

        if files:
            target.files = ", ".join(files)
        if summary.strip():
            target.summary = summary.strip()

        event = "started" if new_status == "doing" else "completed" if new_status == "done" else "updated"
        files_text = target.files if target.files else "-"
        log_line = (
            f"- {now} | request_id: {request_id} | event: {event} | status: {new_status} | title: {target.title} "
            f"| files: {files_text} | note: {summary}"
        )
        self._save_and_verify(self._rebuild(sections, requests, log_line))

    def render(self) -> str:
        self.ensure_exists()
        return self.path.read_text(encoding="utf-8")

    def recover_stale_doing(self, max_age_minutes: int = 90) -> list[str]:
        self.ensure_exists()
        text = self.path.read_text(encoding="utf-8")
        sections = self._split_sections(text)
        requests = self._parse_requests(sections.get("## 요청 목록", []))
        now = datetime.now(tz=timezone.utc)
        threshold = timedelta(minutes=max_age_minutes)
        stale_ids: list[str] = []

        for req in requests:
            if req.status != "doing":
                continue
            ref = req.updated_at_utc if req.updated_at_utc != "-" else req.started_at_utc
            try:
                ref_dt = datetime.fromisoformat(ref.replace("Z", "+00:00"))
            except Exception:
                stale_ids.append(req.request_id)
                continue
            if (now - ref_dt) >= threshold:
                stale_ids.append(req.request_id)

        for request_id in stale_ids:
            self.update_status(
                request_id=request_id,
                new_status="done",
                files=None,
                summary="자동 복구: 이전 세션 중단으로 stale doing 요청을 종료함(필요 시 새 요청으로 재개).",
            )

        return stale_ids

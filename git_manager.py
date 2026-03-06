from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


ARTIFACT_PATHS = [
    "metrics.csv",
    "paper_list.md",
    "work_board.md",
    "STARTUP_CHECKLIST.md",
    "README.md",
    "requirements.txt",
    "docs",
    "blog_Attention_Is_All_You_Need.html",
    ":(glob)blog_*.html",
    ":(glob)papers/**/results/**",
    ":(glob)papers/**/logs/**",
    ":(glob)papers/**/theory.md",
    ":(glob)papers/**/implementation_notes.md",
]


class GitManager:
    def __init__(self, repo_dir: Path, remote_name: str = "origin", branch: str = "main") -> None:
        self.repo_dir = repo_dir
        self.remote_name = remote_name
        self.branch = branch

    def _run(self, *args: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return False, "git command not found"
        msg = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode == 0, msg

    def auto_commit_and_push(self, message: str) -> str:
        ok, msg = self._run("rev-parse", "--is-inside-work-tree")
        if not ok:
            return f"[git] skip: git repository unavailable ({msg})"

        self._run("add", "-A", "--", *ARTIFACT_PATHS)
        ok, status_msg = self._run("status", "--porcelain")
        if not ok:
            return f"[git] skip: unable to read status ({status_msg})"
        if not status_msg.strip():
            return "[git] skip: no changes to commit"

        ok, commit_msg = self._run("commit", "-m", message)
        if not ok:
            return f"[git] commit failed: {commit_msg}"

        ok, push_msg = self._run("push", self.remote_name, self.branch)
        if not ok:
            logger.warning("git push failed; continuing. detail=%s", push_msg)
            return f"[git] commit ok / push skipped-or-failed: {push_msg}"
        return "[git] commit and push completed"

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Config:
    base_dir: Path
    papers_dir: Path
    archive_dir: Path
    metrics_csv: Path
    paper_list_md: Path
    work_board_md: Path
    claude_api_key: str
    github_repo_name: str
    github_pages_base_url: str
    github_pages_dir: Path
    git_remote_name: str
    git_branch: str



def load_config() -> Config:
    base_dir = Path(__file__).resolve().parent
    return Config(
        base_dir=base_dir,
        papers_dir=base_dir / "papers",
        archive_dir=base_dir / "archive",
        metrics_csv=base_dir / "metrics.csv",
        paper_list_md=base_dir / "paper_list.md",
        work_board_md=base_dir / "work_board.md",
        claude_api_key=os.getenv("CLAUDE_API_KEY", "").strip(),
        github_repo_name=os.getenv("GITHUB_REPO_NAME", "").strip(),
        github_pages_base_url=os.getenv("GITHUB_PAGES_BASE_URL", "").strip(),
        github_pages_dir=base_dir / os.getenv("GITHUB_PAGES_DIR", "docs").strip(),
        git_remote_name=os.getenv("GIT_REMOTE_NAME", "origin").strip() or "origin",
        git_branch=os.getenv("GIT_BRANCH", "main").strip() or "main",
    )

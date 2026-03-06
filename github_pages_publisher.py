from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class PublishResult:
    ok: bool
    message: str
    page_url: str = ""
    written_path: str = ""


class GitHubPagesPublisher:
    def __init__(
        self,
        base_dir: Path,
        pages_dir: Path,
        repo_name: str = "",
        pages_base_url: str = "",
    ) -> None:
        self.base_dir = base_dir
        self.pages_dir = pages_dir
        self.repo_name = repo_name.strip()
        self.pages_base_url = pages_base_url.strip().rstrip("/")

    def _guess_base_url(self) -> str:
        if self.pages_base_url:
            return self.pages_base_url
        if self.repo_name and "/" in self.repo_name:
            owner, repo = self.repo_name.split("/", 1)
            return f"https://{owner}.github.io/{repo}"
        return ""

    def _load_index_data(self) -> list[dict[str, str]]:
        data_path = self.pages_dir / "posts_index.json"
        if not data_path.exists():
            return []
        try:
            raw = json.loads(data_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [x for x in raw if isinstance(x, dict)]
        except Exception:
            return []
        return []

    def _save_index_data(self, items: list[dict[str, str]]) -> None:
        data_path = self.pages_dir / "posts_index.json"
        data_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def _render_index_html(self, items: list[dict[str, str]]) -> str:
        grouped: dict[str, list[dict[str, str]]] = {}
        for item in items:
            category = item.get("category", "General") or "General"
            grouped.setdefault(category, []).append(item)

        order = ["LLM", "Diffusion", "Vision", "General"]
        section_blocks: list[str] = []
        for category in order + [k for k in grouped.keys() if k not in order]:
            posts = grouped.get(category, [])
            if not posts:
                continue
            cards: list[str] = []
            for item in posts:
                title = html.escape(item.get("title", "Untitled"))
                href = html.escape(item.get("href", "#"))
                tags = html.escape(item.get("tags", ""))
                updated = html.escape(item.get("updated_utc", "-"))
                summary = html.escape(item.get("summary", "논문 설명과 실험 결과를 정리한 포스트"))
                cards.append(
                    f"""
<article class="card">
  <h3><a href="{href}">{title}</a></h3>
  <p class="summary">{summary}</p>
  <p class="meta"><strong>tags</strong>: {tags}</p>
  <p class="meta"><strong>updated</strong>: {updated}</p>
</article>
""".strip()
                )
            cards_joined = "\n".join(cards)
            section_blocks.append(
                f"""
<section class="section">
  <h2>{html.escape(category)}</h2>
  <div class="grid">
    {cards_joined}
  </div>
</section>
""".strip()
            )

        cards_html = "\n".join(section_blocks) if section_blocks else "<p>아직 게시된 포스트가 없습니다.</p>"
        return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Research Portfolio Blog</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #dbe3ee;
      --accent: #0f4c81;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans KR", "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eef3ff 0%, var(--bg) 40%);
    }}
    .wrap {{
      max-width: 960px;
      margin: 0 auto;
      padding: 32px 16px 48px;
    }}
    .hero {{
      background: #ffffffcc;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 18px;
      backdrop-filter: blur(2px);
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 1.5rem;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .section {{
      margin-bottom: 18px;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 1.1rem;
      color: #123a5c;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }}
    .card h3 {{
      margin: 0 0 8px;
      font-size: 1.05rem;
      line-height: 1.35;
    }}
    .card a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .card a:hover {{
      text-decoration: underline;
    }}
    .summary {{
      margin: 0 0 10px;
      color: #374151;
      font-size: 0.95rem;
    }}
    .meta {{
      margin: 0;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>AI Research Portfolio</h1>
      <p>논문 핵심 개념과 실험 결과를 AI 분야별로 정리한 포스트 모음</p>
    </section>
    <section>
      {cards_html}
    </section>
  </main>
</body>
</html>
"""

    def publish(self, title: str, html: str, tags: list[str], slug: str) -> PublishResult:
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        posts_dir = self.pages_dir / "posts"
        posts_dir.mkdir(parents=True, exist_ok=True)

        post_path = posts_dir / f"{slug}.html"
        post_path.write_text(html, encoding="utf-8")

        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        items = self._load_index_data()
        new_item = {
            "slug": slug,
            "title": title,
            "href": f"posts/{slug}.html",
            "tags": ", ".join(tags),
            "category": tags[3] if len(tags) >= 4 else "General",
            "updated_utc": now,
            "summary": "논문 핵심 개념 설명 + 실험 결과 해석",
        }
        dedup: list[dict[str, str]] = [new_item]
        for item in items:
            if item.get("slug") != slug:
                dedup.append(item)
        self._save_index_data(dedup)
        index_path = self.pages_dir / "index.html"
        index_path.write_text(self._render_index_html(dedup), encoding="utf-8")

        rel = post_path.relative_to(self.base_dir)
        base_url = self._guess_base_url()
        page_url = f"{base_url}/posts/{slug}.html" if base_url else ""
        msg = "[github-pages] publish prepared: commit/push to deploy"
        return PublishResult(ok=True, message=msg, page_url=page_url, written_path=str(rel))

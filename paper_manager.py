from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


class PaperManager:
    def __init__(self, papers_dir: Path, archive_dir: Path) -> None:
        self.papers_dir = papers_dir
        self.archive_dir = archive_dir
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify(name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "_", name.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or "untitled_paper"

    def paper_dir(self, paper_name: str) -> Path:
        return self.papers_dir / self._slugify(paper_name)

    def add_paper(self, paper_name: str) -> Path:
        pdir = self.paper_dir(paper_name)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "results").mkdir(exist_ok=True)
        (pdir / "logs").mkdir(exist_ok=True)

        experiment_code = f'''from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_experiment(paper_name: str, paper_dir: Path, seed: int = 42, iterations: int = 200) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    results_dir = paper_dir / "results"
    logs_dir = paper_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{{run_stamp}}.log"
    logger = logging.getLogger(f"experiment.{{paper_name}}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    logger.info("experiment_start paper=%s seed=%s iterations=%s", paper_name, seed, iterations)

    xs = list(range(iterations))
    ys = []
    iterator = tqdm(xs, desc=f"{{paper_name}} experiment") if iterations >= 100 else xs
    for i in iterator:
        value = max(0.0, min(1.0, 0.3 + (i / max(1, iterations)) * 0.6 + np.random.normal(0, 0.02)))
        ys.append(value)

    score = float(ys[-1]) if ys else 0.0

    fig_path = results_dir / f"metric_curve_{{run_stamp}}.png"
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="score")
    plt.xlabel("iteration")
    plt.ylabel("score")
    plt.title(f"{{paper_name}} metric curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    result_txt = results_dir / f"result_{{run_stamp}}.txt"
    result_txt.write_text(
        "\\n".join(
            [
                f"paper_name: {{paper_name}}",
                f"run_at_utc: {{_utc_now()}}",
                f"seed: {{seed}}",
                f"iterations: {{iterations}}",
                f"final_score: {{score:.6f}}",
                f"figure_path: {{fig_path}}",
            ]
        )
        + "\\n",
        encoding="utf-8",
    )
    logger.info("experiment_done score=%.6f result=%s fig=%s", score, result_txt, fig_path)

    return {{
        "paper_name": paper_name,
        "score": score,
        "note": "auto_experiment",
        "result_path": str(result_txt),
        "log_path": str(log_path),
        "image_path": str(fig_path),
    }}


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    output = run_experiment("{paper_name}", base)
    print(output)
'''
        (pdir / "experiment.py").write_text(experiment_code, encoding="utf-8")
        (pdir / "theory.md").write_text(f"# {paper_name} Theory\n\n핵심 이론 정리\n", encoding="utf-8")
        (pdir / "implementation_notes.md").write_text(
            f"# {paper_name} Implementation Notes\n\n구현 메모\n", encoding="utf-8"
        )
        return pdir

    def remove_paper_to_archive(self, paper_name: str) -> tuple[bool, Path | None]:
        pdir = self.paper_dir(paper_name)
        if not pdir.exists():
            return False, None
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = self.archive_dir / f"{stamp}_{pdir.name}"
        shutil.move(str(pdir), str(target))
        return True, target

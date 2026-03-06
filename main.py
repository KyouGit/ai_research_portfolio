from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
from pathlib import Path

from blog_generator import BlogGenerator
from config import load_config
from git_manager import GitManager
from github_pages_publisher import GitHubPagesPublisher
from md_manager import ChecklistManager, WorkBoardManager
from metrics_tracker import MetricsTracker
from paper_manager import PaperManager


def setup_logging(base_dir: Path) -> None:
    log_dir = base_dir / "system_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def load_experiment_runner(experiment_file: Path):
    spec = importlib.util.spec_from_file_location("experiment_module", experiment_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load experiment module: {experiment_file}")
    module = importlib.util.module_from_spec(spec)
    # Some decorators (e.g. dataclass) expect module to be registered in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "run_experiment"):
        raise AttributeError("experiment.py must define run_experiment")
    return module.run_experiment


def infer_category_from_checklist(checklist_text: str, paper_title: str) -> str:
    current = "General"
    for raw in checklist_text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current = line.replace("## ", "", 1).strip() or "General"
            continue
        if line in {f"- [ ] {paper_title}", f"- [x] {paper_title}"}:
            return current
    return "General"


def ensure_startup_checklist(base_dir: Path) -> Path:
    checklist_path = base_dir / "STARTUP_CHECKLIST.md"
    if checklist_path.exists():
        return checklist_path
    checklist_path.write_text(
        (
            "# STARTUP CHECKLIST\n\n"
            "재시작 시 아래 순서로 컨텍스트를 복구한다.\n\n"
            "1. prom_v2.txt 읽기(상위 경로)\n"
            "2. work_board.md 읽기(doing/todo/done 최신 상태 확인)\n"
            "3. paper_list.md 읽기(현재 학습 우선순위 확인)\n"
            "4. 최신 결과 파일(result_*.txt)과 최신 그래프 확인\n"
            "5. 바로 다음 액션을 request_id와 함께 2~4줄로 선언 후 작업 시작\n\n"
            "주의:\n"
            "- stale doing(장시간 미갱신)은 자동 종료될 수 있다.\n"
            "- OOM 발생 시 자동 확장 금지, 사용자 상담 후 재시작.\n"
        ),
        encoding="utf-8",
    )
    return checklist_path


def is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "hip out of memory",
    )
    return any(marker in text for marker in markers)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.base_dir)
    logger = logging.getLogger("main")

    checklist = ChecklistManager(cfg.paper_list_md)
    papers = PaperManager(cfg.papers_dir, cfg.archive_dir)
    metrics = MetricsTracker(cfg.metrics_csv)
    work_board = WorkBoardManager(cfg.work_board_md)
    git_mgr = GitManager(cfg.base_dir, cfg.git_remote_name, cfg.git_branch)
    blog_gen = BlogGenerator(cfg.claude_api_key)
    gh_pages = GitHubPagesPublisher(
        base_dir=cfg.base_dir,
        pages_dir=cfg.github_pages_dir,
        repo_name=cfg.github_repo_name,
        pages_base_url=cfg.github_pages_base_url,
    )

    checklist.ensure_default()
    work_board.ensure_exists()
    startup_checklist = ensure_startup_checklist(cfg.base_dir)
    stale_ids = work_board.recover_stale_doing(max_age_minutes=90)
    if stale_ids:
        logger.warning("stale doing 자동 종료: %s", ", ".join(stale_ids))
        print(f"[복구] stale doing 자동 종료: {len(stale_ids)}건")
    print(f"[시작] startup checklist: {startup_checklist.name}")

    while True:
        print("\n=== AI Research Portfolio CLI ===")
        print("1. 논문 추가")
        print("2. 논문 체크")
        print("3. 논문 삭제")
        print("4. 실험 실행")
        print("5. 블로그 생성 및 GitHub Pages 반영")
        print("6. 체크리스트 출력")
        print("7. 최근 실험 통계 출력")
        print("8. 작업 보드(work_board.md) 출력")
        print("0. 종료")

        choice = input("메뉴 선택: ").strip()
        if choice == "0":
            print("종료합니다.")
            break

        if choice == "1":
            title = input("논문명: ").strip()
            category = input("카테고리(기본 LLM): ").strip() or "LLM"
            req = work_board.create_request(f"논문 추가: {title}")
            work_board.update_status(req, "doing", ["paper_list.md"], "논문 추가 작업 시작")
            added_to_md = checklist.add_paper(category, title)
            pdir = papers.add_paper(title)
            files = [
                "paper_list.md",
                str(pdir.relative_to(cfg.base_dir) / "experiment.py"),
                str(pdir.relative_to(cfg.base_dir) / "theory.md"),
                str(pdir.relative_to(cfg.base_dir) / "implementation_notes.md"),
            ]
            summary = "논문 목록+구조 생성 완료" if added_to_md else "체크리스트는 기존 항목, 논문 구조만 생성 완료"
            work_board.update_status(req, "done", files, summary)
            print(f"추가 완료: {title} ({summary})")

        elif choice == "2":
            title = input("체크할 논문명: ").strip()
            req = work_board.create_request(f"논문 체크: {title}")
            work_board.update_status(req, "doing", ["paper_list.md"], "체크 상태 변경 시작")
            ok = checklist.check_paper(title)
            work_board.update_status(req, "done", ["paper_list.md"], "체크 완료" if ok else "대상 논문 없음")
            print("체크 완료" if ok else "대상 논문을 찾지 못했습니다.")

        elif choice == "3":
            title = input("삭제할 논문명: ").strip()
            req = work_board.create_request(f"논문 삭제: {title}")
            work_board.update_status(req, "doing", ["paper_list.md"], "삭제 시작")
            removed_md = checklist.remove_paper(title)
            removed_dir, archive_path = papers.remove_paper_to_archive(title)
            changed = ["paper_list.md"]
            if archive_path:
                changed.append(str(archive_path.relative_to(cfg.base_dir)))
            summary = f"md={removed_md}, archived={removed_dir}"
            work_board.update_status(req, "done", changed, summary)
            print(f"삭제 결과: {summary}")

        elif choice == "4":
            title = input("실험할 논문명: ").strip()
            iterations = int(input("반복 횟수(기본 200): ").strip() or "200")
            req = work_board.create_request(f"실험 실행: {title}")
            work_board.update_status(req, "doing", ["metrics.csv"], "실험 실행 시작")

            pdir = papers.paper_dir(title)
            exp_file = pdir / "experiment.py"
            if not exp_file.exists():
                work_board.update_status(req, "done", [], "experiment.py 없음")
                print("해당 논문의 experiment.py가 없습니다. 먼저 논문을 추가하세요.")
                continue

            try:
                run_experiment = load_experiment_runner(exp_file)
                out = run_experiment(title, pdir, seed=42, iterations=iterations)
                record = metrics.append(
                    paper_name=title,
                    score=float(out["score"]),
                    note=str(out.get("note", "auto_experiment")),
                    result_path=Path(out["result_path"]),
                    log_path=Path(out["log_path"]),
                )
                commit_msg = f"auto: experiment {title} score={record.score:.4f}"
                git_result = git_mgr.auto_commit_and_push(commit_msg)
                logger.info(git_result)

                changed_files = [
                    "metrics.csv",
                    str(Path(out["result_path"]).relative_to(cfg.base_dir)),
                    str(Path(out["log_path"]).relative_to(cfg.base_dir)),
                    str(Path(out["image_path"]).relative_to(cfg.base_dir)),
                ]
                work_board.update_status(req, "done", changed_files, f"score={record.score:.4f}")
                print(f"실험 완료. score={record.score:.4f}")
                print(git_result)
            except Exception as exc:
                logger.exception("실험 실행 실패: %s", exc)
                if is_oom_error(exc):
                    message = (
                        "GPU/메모리 한계 감지로 실험을 중단했습니다. "
                        "모델 크기/배치/스텝을 조정할지 사용자와 상담이 필요합니다."
                    )
                    work_board.update_status(req, "done", [], message)
                    print("실험 중단: GPU/메모리 한계(OOM) 감지")
                    print("상담 필요: 더 작은 설정으로 재시도할지 결정해주세요.")
                else:
                    work_board.update_status(req, "done", [], f"실패: {exc}")
                    print(f"실험 실패: {exc}")

        elif choice == "5":
            title = input("블로그 생성 대상 논문명: ").strip()
            req = work_board.create_request(f"블로그 생성/GitHub Pages 반영: {title}")
            work_board.update_status(req, "doing", ["work_board.md"], "블로그 작업 시작")

            pdir = papers.paper_dir(title)
            results = sorted((pdir / "results").glob("metric_curve_*.png"))
            if not results:
                work_board.update_status(req, "done", [], "실험 이미지 없음")
                print("실험 결과 이미지가 없습니다. 먼저 실험을 실행하세요.")
                continue
            image_path = results[-1]
            exp_code_lines = (pdir / "experiment.py").read_text(encoding="utf-8").splitlines()
            max_lines = 120
            code_snippet = "\n".join(exp_code_lines[:max_lines])
            if len(exp_code_lines) > max_lines:
                code_snippet += "\n# ... (truncated for blog readability)"
            score = 0.0
            if cfg.metrics_csv.exists():
                lines = cfg.metrics_csv.read_text(encoding="utf-8").strip().splitlines()
                for line in reversed(lines[1:]):
                    cols = line.split(",")
                    if cols and cols[0] == title:
                        try:
                            score = float(cols[2])
                        except ValueError:
                            score = 0.0
                        break

            slug = papers._slugify(title)
            assets_dir = cfg.github_pages_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            copied_image = assets_dir / f"{slug}_{image_path.name}"
            shutil.copy2(image_path, copied_image)
            image_for_post = Path(f"../assets/{copied_image.name}")

            experiment_context: dict[str, str] = {"score": f"{score:.4f}"}
            experiment_context["category"] = infer_category_from_checklist(checklist.render(), title)
            result_files = sorted((pdir / "results").glob("result_*.txt"))
            if result_files:
                latest_result = result_files[-1]
                for line in latest_result.read_text(encoding="utf-8").splitlines():
                    if ":" not in line:
                        continue
                    key, val = line.split(":", 1)
                    experiment_context[key.strip()] = val.strip()

            draft = blog_gen.generate(
                title,
                score,
                image_for_post,
                code_snippet,
                experiment_context=experiment_context,
            )
            blog_out = cfg.base_dir / f"blog_{slug}.html"
            blog_out.write_text(draft.html, encoding="utf-8")

            publish = gh_pages.publish(draft.title, draft.html, draft.tags, slug)
            changed = [
                str(blog_out.relative_to(cfg.base_dir)),
                publish.written_path,
                str(copied_image.relative_to(cfg.base_dir)),
            ]
            if publish.page_url:
                changed.append(publish.page_url)
            work_board.update_status(req, "done", changed, publish.message)
            print(publish.message)
            if publish.page_url:
                print(f"page_url: {publish.page_url}")
            else:
                print(
                    "page_url 미지정: GITHUB_REPO_NAME 또는 GITHUB_PAGES_BASE_URL 설정 후 "
                    "git commit/push 하면 GitHub Pages에서 확인 가능합니다."
                )

        elif choice == "6":
            print("\n" + checklist.render())

        elif choice == "7":
            print("\n" + metrics.recent_summary(limit=10))

        elif choice == "8":
            print("\n" + work_board.render())

        else:
            print("올바른 메뉴를 선택하세요.")


if __name__ == "__main__":
    main()

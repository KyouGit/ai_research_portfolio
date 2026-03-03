# AI Research Portfolio

Python 3.11 기반 연구 자동화 프로젝트입니다.

## Live Blog
- https://kyougit.github.io/ai_research_portfolio/

## 환경 변수
- `CLAUDE_API_KEY`: Claude API 키 (없으면 블로그 생성은 로컬 템플릿으로 진행)
- `GITHUB_REPO_NAME`: `owner/repo` 형식 (예: `smile/ai_research_portfolio`)
- `GITHUB_PAGES_BASE_URL`: 선택값 (예: `https://owner.github.io/repo`)
- `GITHUB_PAGES_DIR`: 기본 `docs`
- `GIT_REMOTE_NAME`: 기본 `origin`
- `GIT_BRANCH`: 기본 `main`

## 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행
```bash
python main.py
```

## GitHub Pages 설정 (KyouGit/ai_research_portfolio)
아래 값으로 환경 변수를 설정하면 메뉴 `5`에서 생성한 포스트가 `docs/`에 반영됩니다.

```bash
export GITHUB_REPO_NAME="KyouGit/ai_research_portfolio"
export GITHUB_PAGES_BASE_URL="https://kyougit.github.io/ai_research_portfolio"
export GITHUB_PAGES_DIR="docs"
export GIT_REMOTE_NAME="origin"
export GIT_BRANCH="main"
```

리포지토리 연결/푸시:
```bash
git remote -v
git remote set-url origin git@github.com:KyouGit/ai_research_portfolio.git
git add -A
git commit -m "setup: github pages publishing"
git push origin main
```

GitHub 웹 설정(1회):
- `Settings > Pages`
- `Source: Deploy from a branch`
- `Branch: main / docs`

배포 확인:
- CLI 메뉴 `5` 실행 후 `docs/posts/*.html` 생성 확인
- push 이후 `https://kyougit.github.io/ai_research_portfolio` 접속

## 기능
- 논문 추가/체크/삭제(삭제는 `archive/`로 이동)
- 논문별 실험 실행(`tqdm`, `matplotlib`, 고정 seed, PyTorch Transformer 학습)
- WikiText-2 데이터셋 기반 언어모델 실험(첫 실행 시 다운로드 필요)
- 실험 결과 `metrics.csv` 누적
- Git 자동 add/commit/push 시도(실패 시 계속 진행)
- 블로그 HTML 생성 및 GitHub Pages 배포용 `docs/` 반영
- `work_board.md` 요청 추적 자동 동기화

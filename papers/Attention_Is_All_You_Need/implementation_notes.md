# Attention Is All You Need 구현 노트

## 구현 목표
논문 구조를 완전 복제하기보다는, 핵심 원리(Self-Attention + Positional Embedding + Encoder Stack)를 재현 가능한 코드로 구현한다.

## 현재 구현 스펙
- 모델: `TransformerLM` (PyTorch `nn.TransformerEncoder` 기반)
- 데이터셋: `WikiText-2 (wikitext-2-raw-v1)`
- 토크나이즈: 공백 기준 단순 분할
- 학습: `AdamW`, gradient clipping
- 지표: `train_loss`, `validation perplexity`

## 재현성 체크리스트
- seed 고정: `random`, `numpy`, `torch`, `torch.cuda`
- 결과 저장: `results/result_*.txt`
- 그래프 저장: `results/metric_curve_*.png`
- 실행 로그: `logs/run_*.log`
- 메트릭 누적: 프로젝트 루트 `metrics.csv`

## 코드 구조 메모
- `run_experiment()`
  - 데이터 로드
  - vocab 구성
  - 학습 루프
  - 주기적 검증
  - 결과 파일/그래프 저장
- `evaluate()`
  - `CrossEntropyLoss` 평균
  - `perplexity = exp(loss)` 계산

## 추후 개선 아이디어
1. BPE/SentencePiece 토크나이저 도입
2. 학습률 스케줄러(워밍업) 추가
3. mixed precision 학습(fp16/bf16)
4. multi-seed 평균 성능 리포트
5. 실험 설정을 YAML로 분리

## 실험 예산 확장 정책(중요)
- 목표: baseline(unigram/bigram) 대비 Transformer perplexity 개선
- 방법: 성능 미달 시 step/모델 크기 자동 확장 후 재시도
- 시도 정책(예시):
  1. attempt1: d_model=128, layers=2, steps=기본
  2. attempt2: d_model=192, layers=3, steps=2배
  3. attempt3: d_model=256, layers=4, steps=4배
  4. attempt4: d_model=256, layers=6, steps=6배
- 중단 조건: baseline보다 낮은 perplexity 달성 시 즉시 종료
- 결과 기록: 각 attempt별 설정/val_ppl을 result_*.txt에 모두 저장
- 시각화: best attempt 학습곡선 + baseline/transformer perplexity 비교 그래프를 함께 저장

## 참고 이미지
![Transformer Architecture](results/transformer_architecture_diagram.png)
![Attention Formula](results/attention_formula_card.png)

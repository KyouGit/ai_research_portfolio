# BERT Theory

## 핵심 아이디어
- BERT는 문장을 양방향으로 읽는 encoder 기반 사전학습 모델이다.
- 대표 목적함수는 MLM(Masked Language Modeling)으로, 일부 토큰을 가리고 복원한다.
- 토큰을 좌/우 문맥 모두 보며 예측하기 때문에 문맥 표현력이 높다.

## 왜 중요한가
- 검색/분류/질문응답에서 의미 기반 표현(embedding) 품질이 좋아진다.
- 이후의 instruction tuning, retrieval, reranking 기반의 토대가 된다.

## 이번 실험 연결
- unigram baseline(빈도 기반)과 BERT-style MLM을 top-1/top-5 정확도로 비교한다.
- 점수 정의: score = 0.7*top1 + 0.3*top5

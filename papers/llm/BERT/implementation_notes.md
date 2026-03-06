# BERT Implementation Notes

## 실험 목적
- BERT의 핵심인 "masked token 복원"이 단순 빈도 기반보다 나은지 확인한다.

## 구현 개요
- TinyMLM: context token 임베딩 평균 + MLP 분류기
- 입력: 좌우 문맥 토큰(중앙 토큰은 [MASK]로 가정)
- 출력: 마스크된 토큰 분포

## 비교 지표
- baseline_top1_acc / baseline_top5_acc
- bert_top1_acc / bert_top5_acc
- score_gain_vs_baseline

## 해석 포인트
- top1이 오르면 정확 복원이 향상된 것
- top5가 오르면 후보군 생성(검색/추천 초기 단계)에 유리

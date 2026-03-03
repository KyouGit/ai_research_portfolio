# STARTUP CHECKLIST

재시작 시 아래 순서로 컨텍스트를 복구한다.

1. `prom_v2.txt` 읽기(상위 경로)
2. `work_board.md` 읽기(`doing/todo/done` 최신 상태 확인)
3. `paper_list.md` 읽기(현재 학습 우선순위 확인)
4. 최신 결과 파일(`result_*.txt`)과 최신 그래프 확인
5. 바로 다음 액션을 `request_id`와 함께 2~4줄로 선언 후 작업 시작

주의:
- stale doing(장시간 미갱신)은 자동 종료될 수 있다.
- OOM 발생 시 자동 확장 금지, 사용자 상담 후 재시작.

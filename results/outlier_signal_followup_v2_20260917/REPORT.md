# 입력 오류·지속 변화 후속 비교 — 진행 중

기존 입력·checkpoint로 optimizer 0회 진단을 완료했다. 12 source/arm × 2회 smoke = 24 updates를 통과했다. 본학습은 아직 완료되지 않았고 E를 채점하지 않았다.

유일한 계약은 PROTOCOL.md다. B4=3/B5=4 gate scalars이며 dummy parameter는 없다. B5 초기값은 문서에서 미지정되어 B4와 동일한 intercept=-4, 나머지0으로 학습 전에 고정했다.

재개: `.venv/bin/python scripts/run_outlier_signal_followup.py run-all`. 완료 경로는 재사용하고, journal과 epoch checkpoint가 불일치하면 자동 재학습하지 않는다. 최종 과학적 판정은 전체 비교와 검산 후 작성한다.

# LoRA 선행 여부 직접 검증 — 사용자 요청으로 일시정지

**사용자가 다시 시작하라고 명시할 때까지 학습을 재개하지 않는다.** 자동 재개 예약은 없다. 학습 실패나 성능 실패로 판정하지 않는다.

다섯 질문(F0+PLAIN의 이득, F0+MAG의 이득, no-LoRA MAG의 PLAIN 대비 추가 가치, B0/F0 추가 이득 차이, standalone/second-stage 구분)은 아직 최종 평가하지 않았으므로 결론을 내리지 않는다.

- 완료: 선택용 8/16 fits, smoke 8/8 updates.
- 본학습: 8864/16,384 unique updates. 남은 상한: 7520 updates.
- 진행 중이던 경로: `electricity_F0_PLAIN_s81551_lr0.0003`, step 672에서 epoch checkpoint 저장 후 종료.
- 미완료: 부분 진행 1경로, 미시작 7경로. 평가·원점수·효과·상호작용·최종 검산은 미실행.
- 완료 경로의 가중치와 기존 B0/PLAIN/MAG 결과는 보존했다. 기존 학습 재실행 0, joint 0.
- 현재 프로세스 종료, journal과 resume step 일치, 중복 update 0을 PAUSE_VERIFICATION.json에 기록했다.

명시적 재개 요청 후 USER_HOLD.json의 사용자 정지 상태를 해제하고 hash·GPU·journal을 다시 확인해야 한다. 그 후 같은 실행기를 사용하면 완료 fits/smoke를 재사용하고 저장된 epoch 경계부터 남은 업데이트만 실행한다. 추가 LR/seed/rank/gate/데이터는 승인되지 않았다.

ERROR.json/EXECUTION_ERROR.json의 REQUESTED_STOP_EPOCH_BOUNDARY는 이번 사용자 정지 신호에 의한 정상적인 중단 경로다. 성능 실패로 해석하지 않는다. 원자료·가중치·resume은 로컬 캐시에 보존하고 hash만 공개한다.

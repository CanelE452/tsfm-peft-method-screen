# 다음 단계 계약 — v4 이후

2026-09-23. MODEL_READY / DATA_ACCESS_BLOCKED.

## 1. 모델 쪽에서 남은 것

```
pipeline.fit() 에서의 custom module 보존   미검증 (v4 는 wrapper + 공통 trainer 경로로 대체)
training resume parity (optimizer·RNG)     미검증 (이번 scope 는 inference roundtrip)
long-horizon unrolling 경로                명시적으로 제외
group attention 조건화                     범위 밖 (별도 연구축으로 만들지 않음)
STAR §3.3.1-3.4 인과 마스크                원문 재확인 필요
```

## 2. 데이터 계약에서 채워야 할 칸 (전부 미정)

```
제어 명령의 의미·발행/유효/실행값 구분과 당시 확정된 계획의 저장 계약
target measurement 의미·단위·시각
정상 운전과 과도응답의 물리적 정의, TRAIN-only tau0 추정 근거
전환 수, 구간별 날짜 수, 부모 episode/날씨/제어정책 중복과 의존성
H·context·간격과 tau0 의 비율 (step 수가 아니라 물리 시간)
시간순 분할과 정답 겹침 방지. 시뮬레이션 seed 를 독립 건물로 세지 않음
전환 직후/안정/겹친 반응 평가의 원점·날짜 가중
전환 정보가 유용하다는 최소 실용 차이의 근거
FI 로 충분하다는 판정의 근거가 절대 오차인지 업무 허용오차인지 비용인지
```

센서 해상도·업무 허용오차가 없으면 실용 문턱은 미정으로 둔다. 3%/0.02°C 를 복사하지 않는다.

## 3. 예산 — 다시 계산해야 한다

신경망 적응군은 `L0/FI/FM/LI/LM/G/O` 7개다. F0 는 추론, SYS 는 별도.
2LR × 3seed 면 7×2×3 = **42 fits**, 소자료 9회를 더하면 **51 fits** 로 45 가 아니다.
산술 예시일 뿐 실행 승인도 필요량 확정도 아니다. 소자료 반복과 대조의 필요부터 검토한다.

## 4. 진행·중단은 자동 규칙으로 쓰지 않는다

"단순 모델에서 차이 없음 → 정보 없음" 과 "FI 가 사실상 해결 → 중단" 을 자동 규칙으로 두지 않는다.
어느 결과든 판단을 바꾸지 못할 검사는 삭제한다.

## 5. 상태 구분

```
DATA_ACCESS_BLOCKED        현재 여기 (UPSTREAM_DNS_ZONE_UNSERVED — 공개 서비스 zone 이 죽음.
                           망을 바꿔도 안 열린다. 로컬 BOPTEST 승인이 다음 갈림길)
DATA_INSUFFICIENT / RESPONSE_NOT_IDENTIFIED / IMPLEMENTATION_INVALID
NO_EVIDENCE_IN_TESTED_MODELS / INCONCLUSIVE / NO_ADDED_VALUE_IN_TESTED_SCOPE
READY_FOR_NEXT_STAGE
```

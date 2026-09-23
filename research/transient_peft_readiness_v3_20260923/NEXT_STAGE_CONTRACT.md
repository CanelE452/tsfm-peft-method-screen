# 다음 단계 계약 — 채워야 할 칸

2026-09-23. 본학습 계획이 아니다. 데이터가 생겼을 때 **먼저 채워야 할 칸**의 목록이다.

## 1. 지금 바로 필요한 결정 2개

**질문 1. BOPTEST 접근을 어떻게 여는가**
(a) 이 환경의 DNS 가 `api.boptest.net` 을 해석하도록 연다
(b) 다른 환경에서 `GET https://api.boptest.net/testcases` 응답을 받아 전달한다 — 가장 가볍다
(c) Docker 설치를 승인해 로컬 경로를 연다 — 이번 지시문 금지 항목이라 별도 승인 필요

**질문 2. loss 분모 교란을 어떻게 처리하는가**
INPUT arm 이 구조적으로 더 작은 scalar loss 를 받는다(§5). 본실험 전에 정해야 한다.
(a) 모든 arm 에 공통으로 target 행 수로 정규화하는 보정을 적용한다
(b) native loss 를 그대로 두고 INPUT/MOD 비교를 "위치 + 스케일" 복합 효과로만 해석한다
어느 쪽이든 학습률로 몰래 상쇄하지 않는다.

## 2. 데이터 계약에서 채워야 할 칸 (전부 미정)

```
정상 운전과 과도응답의 물리적 정의, TRAIN-only 반응시간(tau0) 근거
제어 명령의 발행 / 유효 / 실행값 구분과 당시 확정된 계획의 저장 계약
전환 수, 구간별 날짜 수, 부모 episode·날씨·제어정책의 중복과 의존성
예측 H · context · 간격과 tau0 의 비율 (step 수가 아니라 물리 시간)
시간순 분할과 정답 겹침 방지. 시뮬레이션 seed 를 독립 건물로 세지 않음
전환 직후 / 안정 / 겹친 반응 평가의 원점·날짜 가중 방식
전환 정보가 유용하다는 최소 실용 차이의 근거
FI 로 충분하다는 판정의 근거가 절대 오차인지 업무 허용오차인지 비용인지
```

센서 해상도·업무 허용오차가 없으면 **실용 문턱은 미정으로 남긴다.**
3%/0.02°C 를 검증된 기준으로 복사하지 않는다.

## 3. 아직 검사하지 않은 구현 항목

```
fit() 진입점의 custom module 보존  이번은 model.forward / peft 직접 경로였다
optimizer state·RNG 복원 후 1 step 일치  학습 재개 가능성을 주장하려면 필요
long-horizon unrolling 경로       이번 지원 범위에서 명시적으로 제외했다
STAR §3.3.1–3.4 인과 마스크       원문 재확인 필요
BOPTEST smoke                    DNS 해석이 열려야 가능
```

## 4. 예산 — 기존 45 fits 를 상속하지 않는다

신경망 적응군은 `L0/FI/FM/LI/LM/G/O` 7개다. F0 는 추론, SYS 는 별도다.
2LR × 3seed 를 유지하면 7×2×3 = **42 fits**, 소자료 9회를 더하면 **51 fits** 로 45 가 아니다.
이는 산술 예시이며 실행 승인도 필요량 확정도 아니다. 소자료 반복과 대조의 필요부터
검토한 뒤 실제 비용을 다시 제안한다.

## 5. 상태 구분

```
DATA_ACCESS_BLOCKED           현재 여기 (DNS_RESOLUTION_FAILED)
DATA_INSUFFICIENT             데이터가 생긴 뒤 판정
RESPONSE_NOT_IDENTIFIED       tau0 추정 후 판정
IMPLEMENTATION_INVALID        이번 검사로 해당 없음 (13 PASS / 0 FAIL)
NO_EVIDENCE_IN_TESTED_MODELS  본비교 후
INCONCLUSIVE / NO_ADDED_VALUE_IN_TESTED_SCOPE / READY_FOR_NEXT_STAGE
```

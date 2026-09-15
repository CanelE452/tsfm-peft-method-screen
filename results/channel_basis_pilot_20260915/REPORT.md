# 채널 공유 파일럿 — 기존 원천을 재사용한 개발 비교

**상태 BLOCKED_GPU_BUSY. CPU toy optimizer 30 updates; 실모델 GPU 구조 점검 0/0 updates/attempts. 본학습 0/0 fits 완료/시도, 0 updates.**

## 실제 실행과 미실행

CPU FP64 구조 환원·순열·초기화·활성 경로, 실제 MOMENT-small hidden512/patch8/head, 공식 SPECIFIC 출력/손실 비교를 수행했다. GPU 준비가 막힌 경우 이 CPU 검사 통과를 GPU 학습 완료로 쓰지 않는다. 시도0은 예측 성능0이나 성능 FAIL이 아니다.

## 무엇을 공유했는가

공개 Time-PEFT의 LoRA/frequency/head를 유지하면서 채널 up을 독립/공유/폭 증가/정적 그룹/정적 기저합으로 바꿨다. 입력 채널 간 새 attention은 없다.

| arm | 채널 블록 | 전체 trainable | 총 모델 |
| --- | ---: | ---: | ---: |
| LORA_HEAD | 0 | 3,317,856 | 38,655,264 |
| SPECIFIC | 8,684,800 | 12,265,312 | 47,602,720 |
| SHARED | 395,008 | 3,975,520 | 39,312,928 |
| SHARED_WIDE | 790,017 | 4,370,529 | 39,707,937 |
| GROUP4 | 789,760 | 4,370,272 | 39,707,680 |
| BASIS4 | 790,016 | 4,370,528 | 39,707,936 |

SHARED_WIDE/BASIS4의 채널 블록 차이는1개, GROUP4/BASIS4는256개다. LORA_HEAD/SHARED/SPECIFIC은 동일 예산 대조군이 아니다. count 절약은 학습 속도나 예측 이득과 별개다.

## 원점수와 같은 예산 비교

신규 V/E 예측 비교가 없어 원점수, 최강 예측 대조군, 계수 활용/경량화 신호는 모두 미판정이다. 빈 지표 표는 미실행을 뜻한다.

## 한계와 종료

C-LoRA 등 채널별 공유·분해는 이미 알려진 원리다. 이번은 Time-PEFT 구조의 통제 변형이며 신규성 확정이나 공개 논문 전체 재현이 아니다. 공식 C-LoRA와 같은 백본 비교, 새 원천 독립 확인, 더 넓은 최적화 비교가 남아 있다. 단일64채널/길이512/단일LR/2seed 개발 비교로 범용성·수렴·새채널 일반화를 주장하지 않는다. 선택1024와 말기 V 감소는 BUDGET_LIMITED로 기록한다.

원래 결과 1290개는 그대로 보존됐다. [검증 기록](independent_verification.json), [고정 계약](contract.json), [선행/환경 차이](../../sources/RELATED_WORK.md). 추가 후보나 자동 후속 학습은 없다.

종료 근거: `ResourceError('BLOCKED_GPU_BUSY')`.

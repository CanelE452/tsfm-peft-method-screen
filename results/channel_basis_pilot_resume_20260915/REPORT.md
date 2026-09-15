# 채널 공유 파일럿 — 기존 원천을 재사용한 개발 비교

**상태 COMPLETE. CPU toy optimizer 30 updates; 실모델 GPU 구조 점검 24/24 updates/attempts. 본학습 24/24 fits 완료/시도, 24576 updates.**

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

| 원천 | seed | arm/role | MSE | MAE |
| --- | --- | --- | ---: | ---: |
| electricity | 40001 | SPECIFIC/selected | 0.757929668 | 0.658715172 |
| traffic | 40001 | SPECIFIC/selected | 1.8449904 | 1.06774286 |
| traffic | 40000 | SHARED/selected | 1.13921621 | 0.766229521 |
| traffic | 40001 | SHARED/selected | 1.40788852 | 0.842386053 |
| traffic | 40001 | SHARED_WIDE/selected | 0.974265796 | 0.715661953 |
| electricity | 40001 | SHARED/selected | 0.392334697 | 0.465642087 |
| electricity | 40000 | SPECIFIC/selected | 0.705647427 | 0.628577761 |
| traffic | 40000 | SHARED_WIDE/selected | 0.946411321 | 0.674687325 |
| traffic | 40000 | LORA_HEAD/selected | 0.621236258 | 0.528570983 |
| electricity | 40001 | LORA_HEAD/selected | 0.33327771 | 0.430750822 |
| electricity | 40000 | SHARED_WIDE/selected | 0.372757895 | 0.450329554 |
| traffic | 40000 | GROUP4/selected | 0.854879112 | 0.629963561 |
| electricity | 40000 | LORA_HEAD/selected | 0.298244707 | 0.398719888 |
| electricity | 40000 | BASIS4/selected | 0.499410769 | 0.525673785 |
| electricity | 40000 | BASIS4/posthoc_coefficient_swap | 0.494859095 | 0.521518384 |
| electricity | 40001 | BASIS4/selected | 0.475112295 | 0.519309616 |
| electricity | 40001 | BASIS4/posthoc_coefficient_swap | 0.466936933 | 0.51274113 |
| traffic | 40001 | BASIS4/selected | 1.33939537 | 0.876087462 |
| traffic | 40001 | BASIS4/posthoc_coefficient_swap | 1.29822782 | 0.861510148 |
| traffic | 40001 | GROUP4/selected | 1.25092855 | 0.843394567 |
| electricity | 40000 | GROUP4/selected | 0.41318478 | 0.478588879 |
| traffic | 40000 | SPECIFIC/selected | 1.47247502 | 0.917191893 |
| traffic | 40001 | LORA_HEAD/selected | 0.70245035 | 0.579125236 |
| electricity | 40001 | GROUP4/selected | 0.420368347 | 0.477845082 |
| electricity | 40001 | SHARED_WIDE/selected | 0.414721104 | 0.484711637 |
| electricity | 40000 | SHARED/selected | 0.383012384 | 0.452997873 |
| traffic | 40000 | BASIS4/selected | 1.2699306 | 0.848716023 |
| traffic | 40000 | BASIS4/posthoc_coefficient_swap | 1.26164681 | 0.847087661 |

평균은 원천별 두seed의 원점수를 먼저 평균한다. MSE는 이미 train-standardized 공간이며 std로 다시 나누지 않았다. [개선율/CI](comparisons.csv), [실제 자원](resources.csv).
- electricity: 최저 MSE LORA_HEAD; 제한된 경량화 True, 계수 활용 False; LORA_HEAD가 더 작고 정확함 True.
- traffic: 최저 MSE LORA_HEAD; 제한된 경량화 True, 계수 활용 False; LORA_HEAD가 더 작고 정확함 True.

## 한계와 종료

C-LoRA 등 채널별 공유·분해는 이미 알려진 원리다. 이번은 Time-PEFT 구조의 통제 변형이며 신규성 확정이나 공개 논문 전체 재현이 아니다. 공식 C-LoRA와 같은 백본 비교, 새 원천 독립 확인, 더 넓은 최적화 비교가 남아 있다. 단일64채널/길이512/단일LR/2seed 개발 비교로 범용성·수렴·새채널 일반화를 주장하지 않는다. 선택1024와 말기 V 감소는 BUDGET_LIMITED로 기록한다.

원래 결과 1360개는 그대로 보존됐다. [검증 기록](independent_verification.json), [고정 계약](contract.json), [선행/환경 차이](../../sources/RELATED_WORK.md). 추가 후보나 자동 후속 학습은 없다.

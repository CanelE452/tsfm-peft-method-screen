# 새 건물의 짧은 이력 적응: 학습량 통제 → 공통 전이 → 소수 계수 적응

작성 2026-09-15. 기존 coverage 가설이나 FAIL/STOP을 변경하지 않는 독립 개발 실험이다. **통과 보장이 아닌, 강한 단순 대조군 이후 추가 가치 확인**이 목적이다. 실행 전 소스·데이터·문서 해시를 봉인한다.

## 데이터와 분리

기존 BuildingsBench v1.0.0 BDG-2 actual hourly electricity 관측 eligibility를 재사용한다. 공식 평가 CSV와 raw/cleaned가 모두 유한하고 절대차≤1e-7인 120일 연속 구간만 사용한다. 이 값은 기존 CSV 정밀도 조건이며 새 결과에 맞춰 변경하지 않는다. 기존 14개 physical building을 전부 제외하고 NFC building_id SHA256 순서의 다음 24개를 source12 / tune4 / dev8로 고정한다. 기존 heldout6의 값·예측은 열지 않는다. 기존 eligibility에는 이전 H3/H14 이력 표준편차 및 달력 조건이 포함되어 있으므로 완전 무선택 모집단이라고 주장하지 않는다.

Source는 첫 56일의 55개 일별 context24→target24 학습 창과 다음 14일 validation을 사용한다. Source validation은 직전 실제 24시간을 사용한 rolling forecast이며, target adaptation의 validation으로 제공하지 않는다. Tune/dev는 건물당 한 forecast origin, H3/H14 두 이력을 사용한다. 각 역할 내 해시 순번에 따라 Wednesday/Saturday를 교대로 배정하고 14일 이후 첫 해당 날짜를 사용한다. 날짜·건물·분산을 점수에 맞춰 바꾸거나 탈락 건물을 대체하지 않는다. Target forecast 값은 학습에 쓰지 않는다. Tune 4개의 미래는 전체 공통 recipe 선택용이고 dev8은 평가용이다. 결과를 본 dev는 이후 독립 test로 재사용할 수 없다.

## 공통 모델·목표·자원

amazon/chronos-2 revision 29ec3766d36d6f73f0696f85560a422f50e8498c. FP32, context24, horizon24, native quantile loss (32 output positions 중 24 supervised; native reduction 유지), batch1, backbone/head frozen, q/k/v/o × 2 attention ×12 blocks =96 modules. LoRA rank1/8, alpha=2r, seed61600, AdamW lr1e-4, wd0, betas .9/.999, eps1e-8, clip1. TF32 off, deterministic algorithms. 동일 seed의 epoch별 창 순열을 사용한다.

주지표는 history 표준편차로 나눈 median RMSE, equal episode macro다. Secondary raw RMSE/MAE, scaled pinball, official-style NRMSE를 모두 보관한다. 모든 예측에 같은 increasing quantile rearrangement를 적용하고 raw도 보존한다. AFFINE은 노출된 H의 일별 F0 예측과 관측으로 양의 기울기 OLS (a≥1e-6); ill-conditioning은 실행 조건 오류로 기록하고 임의 ridge를 추가하지 않는다.

RustDesk `/usr/share/rustdesk/rustdesk`만 기존 사용자 승인 예외. GPU 잠금, 시작 free≥4GiB 30초 안정, update/inference 경계 free≥1GiB 및 비승인 외부 compute 없음. 최대 GPU 대기600초, 총 wall 7200초, RAM free≥2GiB. 공통 자원 중단·수치 오류는 성능 실패가 아니다. 자동 재시도·추가 seed·후보를 금지한다.

## 단계 1–2: 최대 57 fits

1. Local tune: 4 buildings×2 histories×2 ranks=16 fits. 한 trajectory에서 epoch1/4/16과 fixed120 checkpoints를 평가한다. H3는 2/8/32 updates, H14는13/52/208; fixed120은 별도 공통 비교점이다. Tune macro로 **하나의 rank·budget**을 선택한다. 정확한 동률이면 적은 update, 작은 rank 우선이다. 성능을 본 후 LR/rank/budget grid를 늘리지 않는다.
2. Source pooled: 선택 rank로 source12의 총660창을1/2/4 epochs 학습하는1 fit(2640updates); 0 checkpoint 포함 source 미래 validation macro로 하나를 선택한다. 추가 source 데이터·학습 비용은 전이 비용으로 명시한다.
3. Warm tune: pooled initialization으로 tune8 fits; 동일 budget grid에서 하나를 선택한다. Source가 선택한 rank를 공유하므로 warm 쪽 rank 탐색은 별도 하지 않는다. POOLED(무 target update), POOLED_AFFINE(두 target 계수), WARM을 모두 비교한다.
4. Local dev16 fits: 선택 budget과 fixed120을 같은 trajectory에서 저장. 최종 실행 cost는 더 긴 trajectory, 배포 cost는 선택 checkpoint까지의 update로 구분한다. F0/AFFINE은 step0에서 평가한다.
5. Warm dev16 fits: 선택 budget만 실행. Source/tune/dev 간 normalization 공유 없음. Target future로 checkpoint나 arm을 선택하지 않는다.

Tune에서 simple {F0,AFFINE,selected LOCAL} 하나와 transfer {POOLED,POOLED_AFFINE,selected WARM} 하나를 각각 고정한다. **후속 계수 단계 조건**: 선택 transfer가 선택 simple 대비 dev macro ≥1% 개선, building 평균 개선6/8 이상, H3 평균 개선>0, H14 악화≤1%. 이것은 작은 개발 집합의 후속 비용 제한 기준이며 논문 PASS/통계적 확증이 아니다. 미충족 시 STOP_NO_TRANSFER_SIGNAL로 종료한다. 모든 개별 arm 원점수와 episode 성패를 별도 보고하므로 한 episode 악화가 곧 자동 FAIL은 아니다.

## 단계 3: 조건부 최대 36 fits; 총 최대93 fits

조건 통과 시에만 source 각 건물에 동일 rank의 LoRA1 fit(55창×4epochs=220updates), source 미래 validation의0/1/2/4 epochs 중 선택한다. 0보다 좋은 선택이2개 미만이면 종료한다. Source 방향들은 성능 기반 test 선택 없이 source validation으로만 결정한다.

**유일한 추가 방법**: common delta + Σ c_j (source_j delta − common delta), 모든96 modules에 같은 source별 scalar 계수를 공유한다. Backbone/source A/B는 고정하고 c만 학습한다. 초기 c=0은 pooled와 정확히 같아야 한다. 별도 방향 정규화·gate·rank 추가 없음. 계수 AdamW lr.01, 나머지 공통; tune8 fits에서 같은 budget grid로 하나 선택 후 dev16 fits. 출력 AFFINE 및 WARM과 비교한다. 학습 가능한 target parameter 수는 source 방향 수≤12지만 저장 공간·forward/backward 비용은 이 수와 같지 않다.

추가 source12 fits는 baseline 대비 비용이 추가되므로 **동일 총 source compute 조건의 방법 우월성은 주장하지 않는다**. 결과는 shared source 정보 효과와 계수 방식 추가 효과를 분리한다. Best reported fixed control보다 macro ≥0.5% 개선, building6/8, 각 H 악화≤1%이면 DEVELOPMENT_SIGNAL. 이 비교의 best control은 사후 보수적 감사용이며 배포 arm 선택에 쓰지 않는다. 비열등성이나 논문 신규성을 이 작은 실험으로 인정하지 않는다. 조건 미충족 STOP_NO_COEFFICIENT_ADDED_VALUE. 결과가 좋아도 heldout/후속 학습은 자동 실행하지 않는다.

## 한도·검증·해석

최대93 fits, 모델 full optimizer smoke0 updates(합성 layer gradient 검사 별도). 단계1/2 최대11824 updates, 계수 단계 추가 최대6576 updates, 합계 최대18400 updates(선택 budget에 따라 감소). Fit START/COMPLETE/ERROR 원장, 각 step loss/gradient, checkpoint hashes, frozen/buffer 불변, exact reload, raw/sorted 예측 및 독립 float64 metric replay를 남긴다. Source/tune 선택을 독립 검산하고 기존 결과 hashes를 확인한다. 진행 중 오류는 partial artifact와 실행 횟수를 남겨 오류와 성능 실패를 구분한다.

LoRA 일반 개선과 source pooled 개선은 **이미 알려진 접근의 효과**다. 현재 bank 식 자체도 새 기법으로 주장하지 않는다. 신규성 판단의 주요 충돌: [PhysioPFM ICML2025](https://proceedings.mlr.press/v267/wu25ah.html), [LoRA Recycle CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_LoRA_Recycle_Unlocking_Tuning-Free_Few-Shot_Adaptability_in_Visual_Foundation_Models_CVPR_2025_paper.pdf), [Meta-LoRA personalization](https://arxiv.org/abs/2608.12389), [MTA](https://arxiv.org/abs/2511.20072). 향후 방법론이 되려면 같은 정보·예산의 강한 baseline에서 독립 건물·복수 seed의 추가 가치와 이 선행연구와 다른 원리를 입증해야 한다. 이번 실행은 그 전제의 개발 확인이다.

데이터 제한: source/dev는 동일 공개 dataset의 다른 physical buildings이며 완전한 새로운 site/domain 분리는 아니다. 한 origin/건물, single seed, 8 dev buildings이므로 계절·건물 모집단 일반화나 유의성 주장을 하지 않는다. 전처리 원자료 cleaning 및 연속관측 eligibility 선택 편향도 남는다. Raw 데이터·모델·예측 npz/weights는 local ignored cache, GitHub에는 지시문·코드·manifest·원점수·검산 및 한국어 REPORT를 올린다.

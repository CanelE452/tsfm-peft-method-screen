"""Human-reviewed interpretation of completed fixed-contract results; no fitting."""
from pathlib import Path
import hashlib
import json
import shutil
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHORT = {'R_ROLLOUT_LORA': 'R', 'S_STATE_ADAPTER': 'S', 'U_UNCERTAINTY_ADAPTER': 'U',
         'F0_MEDIAN': 'F0-median', 'F0_NATIVE': 'F0-native', 'R_NATIVE': 'R-native',
         'R_MC16': 'R-MC16', 'CHRONOS2_DIRECT': 'C2-direct'}
CATEGORY = 'STATE_OR_CALIBRATION_SUFFICIENT'


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def table(frame):
    frame = frame.copy()
    for col in ['method', 'baseline', 'candidate']:
        if col in frame:
            frame[col] = frame[col].replace(SHORT)
    return '\n```text\n' + frame.to_string(index=False, float_format=lambda x: f'{x:.6f}') + '\n```\n'


def main():
    verification = read('VERIFICATION.json')
    assert verification['status'] == 'PASS'
    assert read('CALIBRATION_INDEPENDENT_AUDIT.json')['status'] == 'PASS'
    extra = read('SUPPLEMENTARY_MODEL_VERIFICATION.json')
    assert extra['status'] == 'PASS' and extra['optimizer_updates_added'] == 0
    ledger_hash = sha(HERE / 'UPDATE_LEDGER.jsonl')
    assert ledger_hash == extra['ledger_sha256_before_after']
    history = HERE / 'SOURCE_HISTORY'
    history.mkdir(exist_ok=True)
    for name in ['REPORT_KO.md', 'FINAL_DECISION.md', 'VERIFICATION.json']:
        original = history / ('automatic_' + name)
        if not original.exists():
            shutil.copyfile(HERE / name, original)

    scores = pd.read_csv(HERE / 'SCORES_REPEAT_MEANS.csv')
    effects = pd.read_csv(HERE / 'EFFECTS.csv')
    seeds = pd.read_csv(HERE / 'SEED_EFFECTS.csv')
    full = effects[(effects.window == 'full_256') & (effects.baseline_variant != 'raw')]
    metric = scores[(scores.window == 'full_256') & (scores.variant != 'raw')]
    metric.to_csv(HERE / 'METHOD_SCORE_TABLE.csv', index=False)
    full.to_csv(HERE / 'PRIMARY_COMPARISON_TABLE.csv', index=False)
    point_cols = ['scaled_mae', 'raw_mae', 'scaled_rmse']
    points = scores[(scores.variant != 'raw') & scores.window.isin(['full_256', 'tail_129_256'])]
    points = points.pivot(index=['source', 'method', 'variant'], columns='window', values=point_cols)
    points.columns = [f'{m}_{w}' for m, w in points.columns]
    points = points.reset_index()
    points.to_csv(HERE / 'POINT_METRICS_FULL_TAIL.csv', index=False)
    resource = pd.read_csv(HERE / 'RESOURCES.csv')
    resources = resource.groupby(['source', 'method', 'variant', 'batch_size'], as_index=False).agg(
        wall_ms=('median_wall_seconds', lambda v: v.mean() * 1000),
        peak_MiB=('peak_allocated_bytes', lambda v: v.max() / (1024 ** 2)),
        loading_ms=('model_loading_seconds', lambda v: v.mean() * 1000))
    resources.to_csv(HERE / 'RESOURCE_SUMMARY.csv', index=False)
    ec = ['source', 'baseline', 'candidate', 'baseline_variant', 'effect_a_minus_b',
          'relative_gain_percent', 'ci95_low', 'ci95_high']
    width = full[full.comparison == 'additional_width_information']
    main_effects = full[full.comparison.isin(['generated_state_information', 'additional_width_information', 'total_U_vs_R'])]
    seed_width = seeds[(seeds.window == 'full_256') & (seeds.baseline_variant != 'raw') & (seeds.comparison == 'additional_width_information')]
    decision = f'''# 최종 판단

실행 상태: COMPLETED_WITH_DISCLOSED_PROTOCOL_DEVIATION

과학적 분류: **{CATEGORY}**

이번 고정 예산에서는 U의 예측 폭 전달이 S보다 실용적으로 의미 있는 추가 가치를 보인다고 판단하기 어렵다. Electricity의 작은 양의 평균 차이는 인정하지만, S가 R 대비 이득의 대부분을 설명하고 U−S는 seed별 방향이 갈린다. ETTh1에서는 U와 S가 거의 같고 둘 다 R보다 전체 확률점수가 나쁘다. 효과가 정확히 0이라는 결론은 아니다.

U의 S 대비 전체256 개선(양수는 U가 좋음):
{table(width[ec])}
일반 R rollout LoRA는 F0 중앙값 경로보다 ordered 점수를 Electricity 2.798%, ETTh1 8.740% 개선했다. 이것은 rollout 학습을 포함한 LoRA 적응 전체의 결과이며, teacher-forcing 대조가 없으므로 rollout 노출만의 인과효과가 아니다.

Chronos-2는 두 자료의 평균 확률점수와 실측 latency에서 강한 대안이다. 그러나 더 많은 메모리, 80% coverage의 차이, ETTh1 점예측 오차와 보정 후 PCE의 손익이 있어 `DIRECT_LONG_MODEL_DOMINATES`라는 전면 우월 분류는 채택하지 않는다. 현재 공식 branching도 중앙값-only보다 개선되므로 이를 약한 중앙값 기준선과 혼동하지 않는다.

24 fits × 512 = 본학습 12,288 updates, 폐기 smoke 12 updates, 합계 12,300으로 종료했다. 26개 CAL/TEST 모델과 104개 자원 조합을 완료했다. 선택 seed 92120은 반복 평균에서 제외했다. 추가 fit·LR·rank·epoch·seed·자료·후속 실험은 0이다.

구현 이탈을 별도 공개한다. 첫 두 Electricity R 선택 fit은 전체 실행 코드 봉인 전에 별도 runner로 실행됐고, 원학습 전후 frozen weights/buffers digest가 측정되지 않았다. 1,024 updates와 원 checkpoint를 보존해 재학습하지 않고 복구했다. 이 두 fit은 LR 선택에 사용됐으므로 완전히 영향 없는 사건이라고 하지 않는다. 나머지 22 fits는 전후 digest를 직접 확인했다. 수치 검산 성공이 이 기록 누락을 소급해 없애지는 않는다. [복구 기록](PRESEAL_EXECUTION_RECOVERY.json), [원 실행 코드](../../experiments/rollout_uncertainty_peft_v1_20260921/FIRST_FIT_RUNNER_ORIGINAL.txt).

자료 실패와 자원 차단은 없었다. 위 구현 사건은 복구됐으며, U의 추가 가치 미확보는 별도의 과학적 관찰이다. 512 updates·두 반복의 제한 아래 판단하며, 신규성이나 논문 PASS를 선언하지 않는다. rollout PEFT 전체의 반증도 아니다. **자동 후속 0으로 종료한다.**
'''
    (HERE / 'FINAL_DECISION.md').write_text(decision, encoding='utf-8')

    text = [f'''# Rollout uncertainty PEFT — 최종 한국어 보고서

**판단: {CATEGORY}. U의 폭 정보 전달에 대한 독자적인 실용적 이득은 이번 범위에서 미확보다.** Electricity에서 U−S의 평균 점수 개선은 약 0.018%이며 두 seed의 방향이 다르고, ETTh1에서는 약 0.001%로 거의 같다. R의 일반 rollout LoRA 학습은 두 자료에서 개선됐으므로 이 결과를 rollout PEFT 전체의 실패로 확대하지 않는다.

2026-09-21 시작, 2026-09-22 KST 종료. 사용자가 준 [MASTER_CLI](../../experiments/rollout_uncertainty_peft_v1_20260921/contract/MASTER_CLI.txt)를 유일한 실행 계약으로 사용했다. HIER·MAG·FR의 모델/결과/학습을 합치거나 재개하지 않았다. 기존 Python 환경만 읽기 전용으로 재사용했고 새 HF cache에서 고정 upstream weights를 로드했다. 아래의 구현 이탈은 숨기지 않는다.

24 fits, main 12,288 + smoke 12 updates를 실제 실행했다. 본학습은 49,152 conditional batch forward/backward 호출이다. batch8의 개별 계열 수나 GPU launch 수와 다른 단위다. 학습 모델의 표 수치는 seed92121/92122 **점수 평균**이며 예측 ensemble이 아니다. 고정 F0와 C2는 한 모델의 수치다. 선택 seed92120은 반복 평균에 없다.

주 지표는 TRAIN 표준편차로 나눈 9개 분위수 mean twice-pinball(낮을수록 좋음)이다. exact CRPS, 순수 calibration 지표, 동시 경로 coverage가 아니다. 이하 개선은 baseline−candidate이며 상대 개선율은 평균 점수 차이를 평균 baseline 점수로 나눈 값이다.

## 1. 현재 공식 branching은 무엇을 해결했는가?

현재 고정한 공식 Chronos-Bolt는 중앙값-only 장기 rollout이 아니다. 첫 9분위수 경로를 확장하고 다음 단계의 9×9 출력을 9분위수로 축약한다. 원래 observed512를 유지하며 실제 호출 context 길이는 512/576/640/704였다. 공식 API를 그대로 호출했고, real TRAIN 입력에서 native64/256 parity 및 호출별 raw 분위수·집계·context retention을 검산했다.

F0-native는 F0-median보다 ordered 전체 점수를 Electricity 1.919%, ETTh1 5.243% 개선했다. 같은 CAL affine 이후에도 각각 0.531%, 2.034% 차이가 남는다. 따라서 현재 공식 방식을 중앙값-only라고 설명하면 비교의 출발점부터 틀린다.
''', table(full[full.comparison == 'official_branching_vs_median'][ec]), '''
선행 [논문](https://arxiv.org/html/2510.16060v2)과 [코드](https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration/tree/b60bfd92ff836525773c71039a83ba2fe3d123bf)를 먼저 확인했다. 그 문헌의 역사적 설명/설정과 현재 공식 구현을 구분했다. 이전 코드의 context 이동 방식과 모델 크기도 이번 계약과 같지 않으므로 정확한 재현이나 신규성 검증을 주장하지 않는다. 공식 코드·모델 revision과 파일 SHA는 [SOURCE_AND_MODEL_MANIFEST.json](SOURCE_AND_MODEL_MANIFEST.json)에 있다.

## 2. 일반 rollout LoRA 학습의 효과는?

R은 자체 생성 중앙값을 학습 중 다음 context에 넣으며, 값과 폭 모두 detach한다. S/U도 같은 방식으로 자기 생성값을 경험한다. R은 F0-median 대비 ordered 점수를 Electricity 2.798%, ETTh1 8.740% 개선했고, 양쪽에 같은 affine 기회를 준 뒤에는 0.822%, 3.310% 개선이다. 학습하지 않은 F0와의 비교이므로 LoRA 적응 및 rollout 노출을 합친 결과다. teacher-forcing 학습 대조 없이 rollout 노출만의 효과라고 부르지 않는다.
''', table(full[full.comparison == 'general_rollout_training'][ec]), '''
## 3. 상태 정보와 예측 폭의 추가 효과는?

공통 q/v rank8 LoRA는 294,912 parameters다. S/U는 동일한 516→8→512 residual adapter 8,744 parameters를 추가하고 zero-up 초기화했다. 같은 seed의 LoRA와 S/U 초기화 hash가 같고 초기 256 예측도 일치했다. S 대 R은 상태 입력과 추가 용량을 함께 바꾸는 비교다. U 대 S는 같은 용량에서 마지막 lead² 채널을 log-width 채널로 바꾼 비교이며, 모든 정보가 동일한 순수 추가 실험은 아니다.

Electricity에서 S가 R 대비 약 0.304%, U가 약 0.321% 개선해 이득 대부분이 S에도 있다. ETTh1에서는 S/U가 R보다 약 0.858%/0.857% 나빠졌다. U−S는 Electricity에서 작은 양의 평균 CI가 있지만, 실용적 크기가 매우 작고 seed별 방향이 반대다. ETTh1은 평균·CI·seed 모두 뚜렷한 추가 신호가 없다. 폭의 proxy를 전달했다고 참 불확실성이나 joint Bayesian propagation을 구현했다고 해석하지 않는다.
''', table(main_effects[ec]), '\nU−S의 개별 반복 효과:\n', table(seed_width[['source', 'seed', 'baseline_variant', 'effect_a_minus_b', 'relative_gain_percent', 'ci95_low', 'ci95_high']]), '''
후반129–256의 U−S ordered 개선율도 Electricity 0.026193%, ETTh1 −0.000084%로 작거나 혼합이다. Electricity의 U PCE 0.031420는 S 0.031203보다 조금 높다. coverage 증가나 점수의 아주 작은 감소를 일괄 calibration 개선이라고 쓰지 않는다.

![블록별 확률점수](FIGURE_1_BLOCK_SCORES.png)

그림 1. 64시간 블록별 score. 오차막대는 두 반복의 범위이며 CI가 아니다. 고정 기준선은 한 모델이다. S/U의 중첩은 실제 차이가 작기 때문이다. PDF도 같은 이름으로 제공한다.

## 4. 단순 출력 보정으로 충분한가?

모든 26개 source/model 조합에 동일한 CAL-only 35점 grid를 블록별 적용했다. 최종 정렬 출력만 보정하며 생성 context에 되먹임하지 않는다. 3,640개 grid 점수를 실제 CAL 정답/예측으로 독립 재계산했고 최대 오차 2.22e−16, winner/tie 모두 일치했다. 보정은 conformal 보장을 하지 않는다.

F0-median의 큰 undercoverage는 간단한 폭 보정으로 상당 부분 줄었다. 반면 Electricity S/U의 affine는 TEST 점수를 각각 약 0.000137/0.000140 악화시켰고 ETTh1 S/U도 악화됐다. CAL 최적이 TEST 최적이라는 보장은 없으며, 이 결과를 보고 계수를 다시 고르지 않았다. 같은 affine 이후 U−S 개선은 Electricity 약 0.015%, ETTh1 약 0.0005%다. 보정이 모든 차이를 정확히 제거했다고 말하지 않는다.
''']
    for source in ['Electricity', 'ETTh1']:
        text += [f'\n{source} 전체256: 학습 모델은 두 반복 점수 평균, F0/C2는 단일 모델:\n', table(metric[metric.source == source][['method', 'variant', 'scaled_pinball', 'pce', 'coverage80', 'scaled_width80']])]
    text += ['''
![보정·coverage·폭](FIGURE_2_CALIBRATION_WIDTH.png)

그림 2. PCE, 80% pointwise coverage, 폭을 별도로 표시했다. 채운 원은 ordered, 빈 사각형은 affine다. 막대는 두 seed 범위이며 CI가 아니다. 0.8 선에 가까워지는 것만으로 probability score의 개선을 뜻하지 않는다.

## 5. 공식 branching·MC16·직접 장기 모델 대비 손익은?

R-native는 같은 선택 R 가중치에 공식 branching을 적용한 대조다. U는 Electricity에서 R-native보다 ordered score 약 0.427% 좋지만 ETTh1에서는 약 0.155% 나쁘다. R-MC16은 두 자료에서 U보다 확률점수가 나쁘고 더 느렸다. 이는 고정 16경로·매 horizon 독립 uniform·q.1/q.9 바깥 clamped tails라는 대조의 결과이며, 모든 stochastic rollout의 실패가 아니다.

Chronos-2의 score는 Electricity 0.173254, ETTh1 0.385099로 U의 0.178534/0.390478보다 낮다. batch8 시간도 약 24/23ms로 U의 약71ms보다 빠르다. 그러나 C2 peak allocated memory는 약563–564MB, U는 약272MB다(아래 표는 MiB 단위). C2 coverage80은 0.7610/0.7515로 U의 0.7975/0.7793보다 0.8에서 멀다. ETTh1 전체256에서 C2의 scaled MAE/RMSE도 U 및 R보다 나쁘고, affine 이후 PCE도 R/U보다 높다. 후반129–256 MAE는 C2가 U보다 조금 낮으므로 전구간 우열로 일반화하지 않는다. 따라서 확률점수·속도의 강한 대안이지만 모든 지표의 전면 지배라고 하지 않는다. ETTh1 U−C2 효과의 원점 CI는 0도 포함한다.
''', table(full[full.comparison.isin(['U_vs_official_F0', 'U_vs_official_R', 'U_vs_MC16', 'U_vs_direct_long_model'])][ec]), f'''
실제 파라미터 수는 Bolt-small {extra['native_audits'][0]['model_parameter_count']:,}, Chronos-2 {extra['chronos2_parameter_count']:,}이다. 크기·구조·사전학습이 달라 순수 adapter 인과 대조가 아니다. Chronos-2는 8개의 독립 group, observed512, output16 patches로 한 번에256을 출력했다. 실제 model forward 한 번, group_ids 0–7, 미래 covariate placeholder의 finite 값 0을 검산했다.

아래는 동일 RTX4070에서 loading을 제외한 end-to-end256 wall time이다. CPU/GPU 전송, metadata, 정렬과 해당 affine를 포함했다. 1회 warmup 후 3회 측정의 중앙값을 모델별 계산하고 두 반복 모델만 평균했다. peak는 seed 중 최대다. Windows 공유 데스크톱 GPU의 짧은 3회 측정이며 독점 GPU/다른 장비의 보편적 속도를 보장하지 않는다. ETTh1 batch8은 독립 7계열 중 한 계열을 한 번 반복해 배치 크기를 맞췄다. loading 시간과 각 측정 min/max는 원 자원표에 있다.
''', table(resources[resources.variant == 'ordered'][['source', 'method', 'batch_size', 'wall_ms', 'peak_MiB', 'loading_ms']]), '''
단일경로/공식 branching/MC16의 예제당 conditional context 수는 4/28/49이며 실제 시간 배수와 다르다. C2 직접 호출은1회다. CPU 입력을 요구하는 C2의 불필요한 CPU→GPU→CPU 복사를 자원 측정 전에 제거했다. 학습 수식은 바꾸지 않았고 원본·diff·hash는 [SOURCE_AMENDMENTS.json](SOURCE_AMENDMENTS.json)에 보존했다.

![품질과 지연시간](FIGURE_3_QUALITY_LATENCY.png)

그림 3. batch8의 예제당 시간은 log축이다. 가로막대는 세 측정 min/max의 seed 평균, 세로막대는 두 seed 점수 범위다. CI가 아니며 memory 손익은 표와 함께 읽어야 한다.

## 6. 자료별·seed별 반례와 검증 범위

전체256과 후반129–256의 점예측 지표를 나란히 보존한다. raw MAE는 각 자료의 원 단위 평균이므로 자료 사이 직접 비교에 쓰지 않는다. ETTh1에는 부하6개와 온도 OT가 섞여 있다.
''', table(points), '''
계열·원점·lead·블록·prefix·raw crossing 상세는 아래 원표에 있다. 원점 stride24, horizon256으로 인접 target이232시간 겹친다. CI는 7연속 원점을 묶은 paired moving-block bootstrap2,000회이며 계열/horizon을 함께 보존한다. 고정된 두 seed와 선택 계열 아래 원점 불확실성만 반영한다. 7일 블록이 256시간 의존을 전부 제거한다고 보장하지 않으며 optimizer 모집단, 다중 비교, 과거 탐색을 보정하지 않는다.

Electricity32계열은 TRAIN 적격 열의 고정 SHA 순서로 선택했다. 타임스탬프 없는 hourly slot이므로 달력/UTC를 검증했다고 하지 않는다. ETTh1은 실제 hourly 시간축을 검증했다. TRAIN/CAL/VAL/TEST는60/10/10/20이며 ETT 표준12/4/4개월 재현이 아니다. scale은 TRAIN만, 미래 정답은 context/metadata에 쓰지 않았다. 공개자료의 사전학습 노출 가능성이 있으므로 독립 확증/미사전학습 benchmark를 주장하지 않는다.

자료 감사·CPU24 tests·실제 Chronos/LoRA GPU smoke·초기 parity·미래치환 불변성·유한 gradient·save/restore·배치/순서 불변성은 통과했다. 양쪽 자료의 96개 VALIDATION checkpoint 점수와 3,640개 CAL grid를 독립 재계산했다. TEST의 실제 소표본 scalar pinball/PCE/coverage와 origin/lead 평균 일치, gain의 baseline−candidate 재계산을 검산했다. 추가 TRAIN-only native 집계 감사도 통과했고 ledger hash는 불변이다.

**구현 사건과 과학적 결과를 구분한다.** 위임 작업이 범위를 넘어 별도 runner를 시작해 첫 두 Electricity R 선택 fit이 전체 코드 봉인 전에 실행됐다. 원본 runner·checkpoint·optimizer/RNG·1,024 spent updates를 보존하고 완료 경계에서 중단한 뒤 재학습 없이 canonical runner로 이어갔다. 두 fit의 원학습 전후 frozen digest는 없다. 복구 모델, 초기/learned checkpoint, frozen 코드 경로를 확인했지만 원학습 buffers 불변을 직접 실측했다고 쓰지 않는다. 나머지22 fits와 smoke는 전후 weights/buffers digest를 직접 확인했다. 이 두 선택 fit은 LR 선택에 기여했으므로 반복 평균에서 제외됐다는 이유만으로 사건의 영향을 무시하지 않는다. 이 제한 때문에 실행 상태는 `COMPLETED_WITH_DISCLOSED_PROTOCOL_DEVIATION`이다.

smoke 기록 시 event 인자 충돌도 저장된 update에서 복구했고, Chronos-2의 CPU DataLoader 입력 연결 문제는 본학습 전에 수정했다. 실패를 성능 실패로 기록하거나 update를 재실행해 예산을 늘리지 않았다. 데이터 실패/자원 차단은 없었다. 장시간 실행 중 처리 속도 변화는 있었으며 비용의 일반화 한계를 함께 보고한다.

모든 24 fits의 설정·선택 step은 [FIT_LEDGER.csv](FIT_LEDGER.csv)에 있다. 512 updates가 충분한 최적화라고 주장하지 않는다. 두 반복은 적고 U−S 효과는 매우 작다. full/tail에서 서로 다른 지표의 작은 방향 변화를 선택적으로 성공으로 묶지 않는다.

## 7. 한정된 후속 투자 근거와 종료

이 고정 파일럿에서는 U의 폭 전달을 독립 연구 후보로 확대할 실용적 근거가 약하다. Electricity의 약한 상태-adapter 이득과 일반 R 학습의 개선은 남지만, 이를 U 고유의 효과로 돌릴 수 없다. 별도 최소효과 문턱이나 모든 CI 양수 규칙을 사후 도입하지 않았다. 판단 범주는 크기·seed 반례·같은 affine 기회·강한 대조·실측 비용을 함께 읽은 수동 검토 결과다.

**STATE_OR_CALIBRATION_SUFFICIENT로 종료한다.** 효과가 정확히0이라는 증명, 신규성/논문 PASS, rollout PEFT 전체의 반증은 아니다. oracle true-context, U width0/셔플, 추가 LR/rank/seed/source 및 자동 후속은 실행하지 않는다.

## 산출물과 재현 범위

- [FINAL_DECISION.md](FINAL_DECISION.md), [VERIFICATION.json](VERIFICATION.json), [SUPPLEMENTARY_MODEL_VERIFICATION.json](SUPPLEMENTARY_MODEL_VERIFICATION.json)
- [데이터·분할](DATA_AND_SPLIT_AUDIT.json), [환경](ENVIRONMENT.json), [버전 lock](requirements-lock.txt), [source/model provenance](SOURCE_AND_MODEL_MANIFEST.json)
- [모델·smoke](MODEL_AND_SMOKE_AUDIT.json), [Electricity VAL 검산](VALIDATION_AUDIT_Electricity.json), [ETTh1 VAL 검산](VALIDATION_AUDIT_ETTh1.json), [CAL 독립 검산](CALIBRATION_INDEPENDENT_AUDIT.json)
- [학습 ledger](FIT_LEDGER.csv), [update ledger](UPDATE_LEDGER.jsonl), [LR 선택](LR_SELECTION.json), [모델 선택](MODEL_SELECTION.json), [CAL 계수](CALIBRATION_PARAMETERS.json), [선택 봉인](ALL_SELECTIONS_SEALED.json)
- [예측 manifest](PREDICTIONS_MANIFEST.json), [원점별](SCORES_BY_ORIGIN.csv), [계열·블록별](SCORES_BY_SERIES_BLOCK.csv), [lead별 압축 CSV](SCORES_BY_LEAD.csv.gz), [raw/ordered/affine summary](SCORES_SUMMARY.csv)
- [반복평균](SCORES_REPEAT_MEANS.csv), [paired effects](EFFECTS.csv), [seed별 effects](SEED_EFFECTS.csv), [full/tail point 지표](POINT_METRICS_FULL_TAIL.csv)
- [원 자원표](RESOURCES.csv), [자원 요약](RESOURCE_SUMMARY.csv), [품질·latency](QUALITY_LATENCY.csv), [그림 manifest](FIGURES_MANIFEST.json), [전체 산출물 manifest](FINAL_ARTIFACT_MANIFEST.json)

원자료, HF weights, 학습 checkpoint와 numerical prediction cache는 로컬 `.cache/rollout_uncertainty_peft_v1_20260921`에 보존하며 GitHub에 올리지 않았다. GitHub에는 hash·경로·설정·코드·집계·검산을 남겼다. GitHub 파일만으로 수치 전체를 즉시 replay할 수 있다고 주장하지 않는다. 이미 사용한 update ledger의 예산을 초기화하거나 `--stage all`을 새 학습 허가로 사용하면 안 된다. 최종 보고서는 본 helper의 수동 판단을 포함하므로 자동 평가 stage만 재실행하면 검토 초안이 다시 생성될 수 있다.
''']
    (HERE / 'REPORT_KO.md').write_text('\n'.join(text), encoding='utf-8')
    fits = pd.read_csv(HERE / 'FIT_LEDGER.csv')
    measured = fits.frozen_digest_scope.eq('before and after fit in this process')
    assert measured.sum() == 22 and (~measured).sum() == 2
    assert sha(HERE / 'UPDATE_LEDGER.jsonl') == ledger_hash
    verification.update(
        scientific_decision=CATEGORY, scientific_decision_method='human review of effects, seed counterexamples, calibration and measured cost; no automatic significance gate',
        execution_status='COMPLETED_WITH_DISCLOSED_PROTOCOL_DEVIATION',
        frozen_backbone_verification={'direct_before_after_main_fits': 22, 'recovery_only_main_fits': 2,
                                     'original_training_digest_missing_fits': fits.loc[~measured, 'fit_id'].tolist(),
                                     'limitation': 'First two selection fits lack original training end digest; recovery evidence is not that measurement.',
                                     'recovery_receipt_sha256': sha(HERE / 'PRESEAL_EXECUTION_RECOVERY.json')},
        additional_audit_sha256={n: sha(HERE / n) for n in ['VALIDATION_AUDIT_Electricity.json', 'VALIDATION_AUDIT_ETTh1.json', 'CALIBRATION_INDEPENDENT_AUDIT.json', 'SUPPLEMENTARY_MODEL_VERIFICATION.json']},
        final_ledger_sha256=ledger_hash,
        figures_visual_review='All three rendered PNGs inspected for legends, labels, clipping and matching data; ranges explicitly distinguished from bootstrap CI.',
        failures={'data': False, 'resource_block': False, 'implementation_orchestration_recovered': True},
        automatic_followup_experiments=0,
        final_report_script_sha256=sha(__file__),
        reports_sha256={n: sha(HERE / n) for n in ['REPORT_KO.md', 'FINAL_DECISION.md']})
    (HERE / 'VERIFICATION.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    final = {'execution_status': verification['execution_status'], 'scientific_decision': CATEGORY,
             'main_updates': 12288, 'smoke_updates': 12, 'fits': 24, 'automatic_followup_experiments': 0,
             'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': 'complete',
             'limitations': ['first two selection fits lack original training frozen digest', 'two fixed repeat seeds', '512 updates per fit']}
    (HERE / 'PROGRESS.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(CATEGORY, 'reports written; no optimizer calls')


if __name__ == '__main__':
    main()

"""Record the truthful bounded audit outcome; does not launch experiments."""
import csv
import json
import subprocess
from pathlib import Path
from .audit import ROOT, OUT, CACHE, save, sha, csvwrite


def main():
    source = ROOT/'experiments/outlier_signal_peft_v1_20260917'
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol = sha(OUT/'PROTOCOL.md')
    save('MACHINE_CONTRACT.json', dict(
        status='PREPARATION_ONLY_NOT_TRAINING_SEAL', sole_protocol_sha256=protocol,
        source_order=['electricity','ettm1'], arms=['A0','A1','A2','A3','A4','A5'],
        context=512, horizon=64, channels=4, learning_rates=[.0001,.0003],
        selection_seed=81500, repeat_seeds=[81501,81502], updates_per_fit=1024,
        checkpoints=[0,256,512,768,1024], max_fits=48, max_main_updates=49152,
        max_smoke_updates=24, hard_wall_hours=24, new_cache_limit_gib=50,
        effective_batch=32, microbatch=None, dtype='FP32', TF32=False,
        optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0,grad_clip=1),
        blocked=['supplied reference files absent'],
        unsealed=['supplied component compatibility','condition generator implementation','microbatch','training runner'],
        no_automatic_followups=True))
    save('SOURCE_MANIFEST.json', dict(
        base_commit='72eda9650a121d484f5246da725d4b6a6ed5733a', inspected_head=head,
        remote_head_at_audit='72eda9650a121d484f5246da725d4b6a6ed5733a',
        initial_tracked_worktree_clean=True, duplicate_exact_completed_comparison=False,
        protocol_download='/home/minjae/Downloads/outlier_signal_peft_cli_20260917.txt',
        sole_protocol_sha256=protocol,
        supplied_reference_files={'reference_core.py':'NOT_FOUND','test_reference.py':'NOT_FOUND'},
        search_scope=['Downloads file tree and all ZIP central directories','Documents','Codex attachments','home file-name search'],
        ignored_false_match='Anaconda fsspec test_reference.py is unrelated',
        upstream=json.loads((OUT/'upstream_receipts.json').read_text()),
        implementation_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(source.glob('*.py'))}))
    ledger=[]
    for source_name in ['electricity','ettm1']:
        for arm in ['A0','A1','A2','A3','A4','A5']:
            for lr in [1e-4,3e-4]:
                ledger.append(dict(source=source_name,arm=arm,phase='select',seed=81500,lr=lr,status='NOT_STARTED',updates=0,reason='BLOCKED_MISSING_REFERENCE_FILES'))
            for seed in [81501,81502]:
                ledger.append(dict(source=source_name,arm=arm,phase='repeat',seed=seed,lr='NOT_SELECTED',status='NOT_STARTED',updates=0,reason='BLOCKED_MISSING_REFERENCE_FILES'))
    csvwrite('FIT_LEDGER.csv',ledger)
    save('status.json',dict(execution='PARTIAL',reason='BLOCKED_MISSING_REFERENCE_FILES',
                            completed_fits=0,main_optimizer_updates=0,smoke_optimizer_updates=0,
                            gpu_learning_started=False,cpu_model_audit_completed=True,
                            supplied_reference_tests_run=False,local_tests_passed=5,
                            scientific_decision='NOT_EVALUATED',
                            next_input='Local path to supplied ZIP or reference_core.py and test_reference.py',
                            automatic_background_worker=False))
    (OUT/'literature_boundary.md').write_text('''# 실제 선행 코드 경계

- [TSFM-Biases](https://github.com/amazon-science/TSFM-Biases), commit `3d526e0aaeb513515928c825e6f054b1228f2a5a`의 `notebooks/outlier-bias.ipynb`를 읽었다. blob `723583c1be813c54d90886e73837a543d1772580`은 계약과 일치한다. 주기 160의 cos(t+0.3), 문맥 512·미래 64, 공개 Chronos-T5-small/Bolt-small 연결과 patch-size-1 자리표시자를 확인했다. 입력 표현과 오류 민감성에 대한 직접 선행이며 새 PEFT 효과의 증거는 아니다.
- [TATO](https://github.com/thulab/TATO), commit `402bbc8998c49e2f33d9afbcc42140347a6b8c36`의 `transformation/library/inputer.py`, `model/model_factory.py`, `experiment/run.py`를 읽었다. Inputer는 none/3_sigma/1.5_iqr 검출과 선형 보간을 사용한다. Chronos 연결은 ChronosPipeline, FP16, 3 samples이며 Chronos2 연결도 별도로 있다. 검색 뒤 validation이라고 적힌 구간이 해당 코드에서 train split을 다시 만드는 점도 확인했다. 이번 고정 clip/Hampel 대조는 공식 검색이나 전체 TATO 재현이 아니다. TATO보다 우수하다는 결론은 낼 수 없다.
- [Chronos-Bolt 모델](https://huggingface.co/amazon/chronos-bolt-small)의 고정 snapshot과 설치된 chronos-forecasting 2.3.2 실제 encode/forward를 읽고 CPU에 연결했다. 설치 소스 SHA256은 MODEL_RECEIPT.json에 기록했다. native loss는 정규화된 target을 사용하므로 이번 TRAIN 고정 scale loss와 같다고 가정하지 않았다.

새 경로는 강건 전처리·residual adapter·bounded embedding과 가깝다. 신규성은 확보하지 않았다. 실제 품질 레이블, 정식 선행 비교, 독립 원천 검증은 본 파일럿 이후에도 남는 별도 과제다. 이를 자동 실행하지 않는다.
''')
    (OUT/'reproduction_differences.md').write_text('''# 제한 재현의 경계 — 아직 미실행

원문 notebook의 magnitude sweep은 고정 네 위치와 계수, 100 amplitudes를 사용하며 일부 경로는 미공개 patch-size-1 모델을 요구한다. count sweep의 난수 seed도 고정되어 있지 않다. 계약은 공개 Bolt/T5 두 모델 각각 reference 1 + 2 counts × 3 amplitudes × 8 seeds = 49 series, 합계 최대 98 series로 제한한다. T5는 원문 일부 호출의 1 sample 대신 20 samples의 median이다.

이는 reduced replication이며 Figure 8 전체 재현이 아니다. patch-size-1은 SKIPPED_UNAVAILABLE_REFERENCE로 남긴다. 모델 다운로드 및 실제 코드 확인만 완료했고, 제한 재현 forward 0/98이다. 실제 모델 CPU 연결 검사에서 사용한 Electricity 8개 입력의 forward는 이 98 series에 포함하지 않는다.
''')
    (OUT/'REPORT.md').write_text('''# 입력 오류 강건성·지속 변화 보존 — 부분 진행 보고

**상태: PARTIAL / BLOCKED_MISSING_REFERENCE_FILES. 학습 0/48 fits, optimizer update 0. 성능 실패 판정이 아니다.**

## 완료한 실제 작업

기준 및 원격 main은 `72eda9650a121d484f5246da725d4b6a6ed5733a`였다. 정확히 같은 완료 실험은 없었다. 다운로드의 `outlier_signal_peft_cli_20260917.txt`를 유일한 계약으로 복사했으며 이전 실험을 재개하지 않았다.

공개 두 모델의 revision과 가중치 hash, TSFM-Biases와 TATO의 실제 코드를 확인했다. 두 원자료 hash를 확인하고 TRAIN만으로 첫 네 적격 채널 및 population std를 고정했다. ETTm1 날짜는 단조·중복 없음·15분 격자를 통과했다. Electricity에는 원본 timestamp가 없어 index-day를 쓴다.

| 원천 | 채널 | TRAIN/V/E 서로 다른 날짜 | 선택 날짜 span: TRAIN/V/E |
|---|---|---|---|
| Electricity | 원본 열 0/1/2/3 | 256/64/128 | 630/214/216일 |
| ETTm1 | HUFL/HULL/MUFL/MULL | 256/64/128 | 428/142/144일 |

각 역할의 full eligible day에서 날짜를 먼저 고르고 phase roster를 한 번 섞었다. 모든 phase count max-min≤1이다. day/week/phase/overlap/correlation 원표는 `origin_audit.json`에 있다. 입력과 정답은 별도 캐시에 보관했다. E 정답은 complete-case 검사와 분리 저장에만 접근했고 예측 성능을 채점하지 않았다. 이 원천의 개발 기간은 기존에 사용했으므로 독립 test라고 부르지 않는다.

실제 Bolt-small의 CPU native forward와 명시적 wrapper를 연결했다. q/v 36개에 LoRA rank8/alpha16을 붙였다. A0/A2의 학습 파라미터는 294,912개, A4/A5는 299,784개다. 실제 Electricity 8개 입력에서 native-wrapper 및 A4/A5 초기 출력과 A2의 TRAIN-scale 정규화 최대 차이는 모두 0이었다. A4/A5 초기 추가 가중치 hash가 같고 원래 본체와 head hash가 보존됐다. 이는 초기 연결 검사이며 학습·복원 검사를 대신하지 않는다.

별도로 작성한 `test_local_components.py`의 CPU 검사 5개를 실행해 통과했다. 실제 입력 8개에 과거 오류를 가한 clip/Hampel/feature의 NumPy-Torch 대조, scalar 2-pinball 및 단위 불변성, full-day 선정·부족 날짜 거부, adapter bound와 초기 동일성을 확인했다. **이 파일은 첨부된 test_reference.py가 아니며 그 검사를 통과했다고 주장하지 않는다.**

## 멈춘 조건과 정확한 미실행 범위

사용자가 지정한 `reference_core.py`, `test_reference.py`를 Downloads·Documents·Codex attachments 및 홈 파일명 검색에서 찾지 못했다. Downloads의 ZIP 내부 파일 목록에도 없다. 동명 fsspec 테스트는 무관한 라이브러리 파일이다. 사용자 지시의 “reference_core.py와 test_reference.py를 검토하고 CPU 검사를 실행해”를 아직 충족하지 못했다. 자체 구현으로 해당 첨부 검사까지 통과한 것으로 바꾸지 않았다. 두 파일 또는 ZIP의 로컬 경로가 필요하다.

아직 전체 학습 runner를 완성한 상태가 아니다. generator/label stream 권한 검사, 첨부 코드 대조, GPU microbatch 봉인, 24 smoke updates, 선택 24 fits, 반복 24 fits, 최대 98 series 제한 재현, 선택 봉인, E 예측·채점, 선택 모델 자원 측정·독립 검산 모두 미실행이다. 본학습 49,152 updates와 smoke 24 updates 예산은 전부 남아 있다. `FIT_LEDGER.csv`의 48행은 NOT_STARTED 계획 슬롯이며 실행 횟수가 아니다. GLOBAL_EVALUATION_SEAL·선택·점수 파일을 가짜로 만들지 않았다.

현재 실행 중인 학습·백그라운드 worker는 없다. GPU 조회 당시 RTX3080의 여유 메모리는 9,086 MiB였고 외부 compute는 허용된 RustDesk 272 MiB뿐이었다. GPU 안전 자체가 이번 차단 원인은 아니다. CPU 검사가 사용한 자원과 향후 GPU 학습 비용을 혼동하지 않는다. 학습 peak/time 및 추론 자원 절충은 미측정이다.

## 과학적 질문과 아직 판단할 수 없는 것

측정 오류는 입력만 바꾸고 지속 변화는 과거 끝과 미래를 함께 바꾸어, 오류 감소와 유용한 변화 보존을 분리하려는 비교다. 실제 오류·사건의 전문가 레이블은 확보하지 않았다. 원자료는 UNMODIFIED_REFERENCE이고, SYNTHETIC_MEASUREMENT_FAULT / SYNTHETIC_PERSISTENT_SHIFT / HISTORY_TRIGGERED_SUBSET은 서로 다르다. 동일한 관측 과거에 다른 정답이 있을 때 이를 oracle scenario로 구분해서는 안 된다.

A1-A0는 증강, A2/A3-A1은 전처리, A5-A4는 같은 크기 adapter에서 제거된 관측 차이의 추가 가치, A5-A2는 추가 경로 전체 가치를 분리할 예정이다. 아직 비교 점수가 없으므로 오류 감소·원자료 성능·지속 변화 보존·추가 비용 모두 N/A다. 단순 방법의 충분성이나 새 구성요소의 필요성도 판단하지 않았다. 좋은 결과라도 논문 PASS로 부르지 않는다.

## 재개

첨부 두 파일을 확보해 hash와 CPU 검사 및 현재 구현 대조부터 진행한다. 현재 원점·모델 캐시는 hash 검증 후 재사용하며 중복 준비·다운로드를 피한다. 현 시점의 실행 가능 명령은 CPU 감사와 자체 검사뿐이다:

```bash
.venv/bin/python -m experiments.outlier_signal_peft_v1_20260917.audit
.venv/bin/python -m unittest experiments.outlier_signal_peft_v1_20260917.test_local_components -v
```

이 명령은 학습을 시작하지 않는다. 전체 runner 구현·봉인·실제 GPU correctness 이후에만 승인된 48 fits 범위까지 진행할 수 있다. 원자료·가중치·입력/정답 캐시는 GitHub에 올리지 않으므로 GitHub만으로 수치 재실행이 완결되는 것은 아니다.
''')
    (OUT/'FINAL_DECISION.md').write_text('''# 현재 결정 — 입력 미충족으로 과학적 판정 보류

- 실행: PARTIAL / BLOCKED_MISSING_REFERENCE_FILES. 완료 학습 0/48, main/smoke optimizer 0/0.
- 효과: NOT_EVALUATED. 원점수·오류 감소·지속 변화 보존·자원 이득을 주장하지 않는다.
- 단순 대안의 충분성 및 후보 추가 가치: 아직 비교하지 않았으므로 보류.
- 신규성: 미확보. 공개 강건 전처리·증강·residual adapter 선행과 정식 비교가 남는다.
- 다음 투자 후보: 이번 미실행 감사로는 0개. 이는 방법의 성능 FAIL 또는 연구 중단 판정이 아니다.
- 보존: 데이터 원점/scale/모델 고정 receipt와 CPU 초기 연결 구현. 첨부 reference 파일 확인 뒤 같은 승인 범위를 재개할 수 있다.
- 추가 후보·데이터·seed·LR·다른 연구 및 자동 후속 학습은 실행하지 않았다.
''')
    save('verification.json',dict(scope='preparation_and_initial_CPU_connection_only',
                                  local_cpu_tests=5,local_cpu_tests_passed=5,
                                  supplied_reference_tests='NOT_RUN_MISSING_INPUT',
                                  actual_training_verification='NOT_RUN',
                                  evaluation_scalar_verification='NOT_RUN',
                                  original_results_modified=False,
                                  passed_audit_artifacts={p.name:sha(p) for p in
                                      [OUT/'DATA_MANIFEST.json',OUT/'origins.csv',OUT/'origin_audit.json',
                                       OUT/'MODEL_RECEIPT.json',OUT/'PARAMETER_RECEIPT.json',OUT/'implementation_checks.json']}))


if __name__ == '__main__': main()

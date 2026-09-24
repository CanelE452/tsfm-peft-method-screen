# Transient PEFT — 직접 작성한 고정 실행 패키지 v5

## 목적과 범위

좁은 설비 도메인의 전환 이력을 활용하는 PEFT를 시험하기 전에, 기존 v4의 **학습 객체와 저장 객체 불일치**, **갱신 도중 수동 난수 주입**, 그리고 기존 BOPTEST 분석의 **변화량/잔차 혼동**을 닫는 코드다. CLI가 새 Python 코드를 작성하거나 기존 파일을 고치는 작업은 없다. 연구 타깃·예측 길이·시뮬레이션 생성·본학습은 이번 범위 밖이다.

기준 저장소: `CanelE452/tsfm-peft-method-screen`
읽은 기준 커밋: `7cbe663528602136badd6be3779f0d5e7dd64b14`.

## 한 번 실행

ZIP을 저장소 **밖**에 압축 해제한다. 실제 대상 저장소 루트에서 다음 명령을 한 번 실행한다.

```bash
python3 /실제/압축해제경로/transient_owned_v5_20260924/execute.py --repo "$PWD" --publish
```

`/실제/압축해제경로`만 실제 위치로 바꾼다. 나머지 설정·코드·학습률·시드·검사 횟수는 바꾸지 않는다. 이미 실행한 패키지를 다시 실행하면 중단한다. 실패하면 코드를 고치지 않고 receipt와 로그를 반환한다.

기존 `.venv/bin/python`만 사용한다. 설치된 `chronos-forecasting==2.3.2`, `peft==0.18.1`을 확인하며 버전이 다르면 설치/다운그레이드하지 않고 실패를 남긴다. `amazon/chronos-2`는 기존 캐시에서만 찾아서 snapshot을 고정한다. 새 모델 다운로드·Docker/API 호출·GPU 사용은 없다.

## 구현상 바꾼 점

* 모델이 단일 `response` 모듈을 소유한다. 반응시간 파라미터는 도메인 전체에 공유하고, 상태값만 과제별 명령에서 계산한다. 학습 루프에서 과제별 외부 생성기를 만들지 않는다.
* 모든 attention q/k/v/o에 표준 LoRA를 적용한다. 상태에 의한 조절은 **시간 attention만** 사용한다. 그룹 attention은 일반 LoRA다.
* LoRA를 작은 명시적 PyTorch 모듈로 구현했다. `W(x) + (alpha/r) B(A(x))`, bias 추가 없음, dropout 0, A Kaiming-uniform, B=0이다. Ubuntu 실행에서 설치된 PEFT의 독립 Linear와 비영 A/B의 출력 및 gradient를 대조한다. 이 대조가 실패하면 native 실행으로 넘어가지 않는다.
* `forward` 안에서 raw 명령 → 상태 → 입력 행 또는 modulation을 만든다. 모든 구성은 같은 target history, raw command history, 미래 확정 명령 계획을 받는다.
* 조건부 값의 모양이 다르면 오류다. 조용한 건너뛰기는 없다. REG와 명령을 관측하지 못한 패치에는 명시적인 중립 마스크를 곱한다.
* **현재 버전의 INPUT(FI/LI)은 command gap을 명시적으로 거부한다.** 관측되지 않은 값을 0으로 채우고 native 관측값 전용 정규화와 같다고 주장하지 않는다. MOD의 gap 검사는 별도로 지원한다. 모든 lifecycle fixture는 완전히 알려진 raw command/plan을 사용한다.
* native 분위수 손실의 horizon 평균·quantile 합은 유지하고 마지막 행 평균만 유효 target 행 기준으로 보정한다. padding된 horizon 분모를 변경하지 않는다. 고정 예측값·정답·마스크의 값과 gradient를 검산한다.

## 고정 검사 예산과 실패 처리

L0/FI/FM/LI/LM 각각 **6회**, 최대 **30회 optimizer update**다. 이전 3회 상한을 6회로 늘린 이유는 0 초기화된 B → modulator → tau 경로의 단계적 활성화를 관찰하기 위해서다. 수렴이나 성능 평가가 아니다. 학습률은 처음부터 끝까지 `0.001` 하나다. 갱신 도중 난수 추가·학습률 변경·성공할 때까지 반복은 없다. 여전히 변화가 관찰되지 않으면 `INCONCLUSIVE_UPDATE`로 반환한다.

CPU FP32 native 연산, 공유 log_tau는 작은 변화의 기록을 위해 FP64로 소유하고 상태 출력은 미분 가능한 cast로 FP32에 전달한다. 이를 물리 시상수 측정으로 해석하지 않는다. eval 모드로 dropout을 끈 상태에서 gradient/optimizer만 사용한다. 실제 대규모 `pipeline.fit()` 재현이나 training resume를 주장하지 않는다.

각 stage는 독립 프로세스에서 한 번만 실행한다. 프로세스당 20분·8 GiB RSS, 전체 60분 상한이다. 사용 가능 RAM 4 GiB 미만이면 해당 stage를 실행하지 않는다. 그 stage만 중단하고 예산·오류를 남긴다. 원래 실행 중인 다른 프로세스는 종료하지 않는다. 단위검사/작은 algebra 연산과 native 모델 호출 원장은 구분한다.

## 학습 → 저장 → 새 프로세스 복원

실제로 갱신한 **같은 모델 객체**에서 LoRA A/B, 상태 생성기, modulator, 고정 buffer, 설정을 저장한다. compact checkpoint는 정확한 key 집합·형상·dtype으로 검사하며 `strict=False`로 누락을 덮지 않는다. 새 프로세스에서 원래 캐시 snapshot을 다시 읽고, frozen 원 파라미터 해시와 compact state 해시, 상태값·modulation·예측값을 비교한다. 모든 구성의 최종 학습 checkpoint가 대상이다. 별도 새 모델에 값을 수동 주입한 roundtrip으로 대체하지 않는다.

**학습 재개(resume)는 이번 주장 밖**이다. optimizer/RNG 복원 후 추가 1-step 일치는 주장하지 않는다.

## 기존 BOPTEST 자료 검산

이미 읽은 `results/boptest_data_contract_20260924/`의 CSV/JSON만 읽는다. 기준 커밋에서 바뀌었으면 자동 해석하지 않고 멈춘다.

1. 기존 tau를 고정한 지수 적합의 **실제 잔차 `관측값 - 적합값`**를 계산한다. 새 tau를 찾거나 모드를 식별하지 않는다.
2. `slow_part_span_20min`이 ON 구간 첫/끝 값 차이와 일치하는지 확인하고, 변화량과 잔차를 분리한다.
3. 같은 30초 궤적을 여러 간격·모든 격자 위상으로 성기게 선택한다. 선형보간과 이전 표본 유지의 차이를 별도 CSV로 출력한다. 이는 offline 재구성 비교이며 모델 예측 오차가 아니다.
4. 10초/30초 CSV는 별도 실행으로 표시하고 초기 상태 차이를 원인으로 확정하지 않는다.

새 그림이나 발표 자료를 자동 제작하지 않는다. 결과 JSON/CSV를 여기서 다시 읽어 분석한다.

## 결과 경로와 Git

설치 코드: `experiments/transient_owned_v5_20260924/`
공개 결과: `results/transient_owned_v5_20260924/run_<UTC>/`
비공개 checkpoint: `runs/transient_owned_v5_20260924/run_<UTC>/` (Git ignore)

`--publish`는 **이 패키지 코드와 이 실행의 작은 결과 파일만** commit·push하라는 명시적 승인이다. 실패 결과도 같은 방식으로 게시한다. 기존 v3/v4 파일·원자료·체크포인트·무관한 dirty 파일은 수정하거나 stage하지 않는다. 패키지 checksum이 바뀌었으면 publish하지 않는다. 원격/로컬 HEAD 불일치, 다른 staged 변경, 인증 실패에서는 pull/rebase/reset/force push로 해결하지 않고 중단한다. 성공 후 원격 SHA를 확인하며, 최종 receipt는 ZIP의 압축해제 폴더 바깥에 남긴다.

## 결과의 의미

`NATIVE_FIXTURE_LIFECYCLE_VERIFIED`: 고정된 native fixture에서 구현 생명주기가 확인됨.
`TEST_DOUBLE_ONLY_VERIFIED`: 작성 환경의 작은 대체 모델 검사만 완료. native 검증이 아님.
`READINESS_NOT_CONFIRMED`: 실패 또는 미확인 항목 있음. CLI에서 수정/튜닝하지 않음.

어느 상태도 **과도응답이 실제 예측 difficulty라는 증거, PEFT 성능 개선, 신규성, 실데이터 일반화**를 뜻하지 않는다.

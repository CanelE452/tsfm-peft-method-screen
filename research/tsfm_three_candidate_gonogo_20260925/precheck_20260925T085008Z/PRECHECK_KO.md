# TSFM 세 후보 P0 사전점검

**P0 완료 / 실행 입력 미준비.** 모델 학습·추론·설치·다운로드·TEST 채점은 모두 0회다. 문서의 10개 학습 작업은 후속 설계 예산이며 이번 실행 승인이 아니다.

## 저장소와 범위

- 실제 저장소: `/home/minjae/Documents/github/tsfm-peft-method-screen`
- 기준 HEAD / 원격 main: `af49ead03f31fbe35d51f5ab3fb2295a09fe0305`
- branch: `main`; 시작 시 staged/dirty 변경 없음.
- 푸시 대상: `CanelE452/tsfm-peft-method-screen`의 `main`. 사용자의 고정 게시 지시를 우선한다.
- 두 첨부 문서의 hash와 실제 확인 범위는 PRECHECK.json에 기록했다. 현재 작업은 P0의 읽기·메타데이터 확인·PRECHECK 두 파일 작성에 한정했다.

## 후보별 준비 상태

| 후보 | 실제 source | snapshot | data | 평가표본 지원 | 재사용 cache | blocker |
|---|---|---|---|---|---|---|
| A 공유 LoRA | MSFT 미발견 | Moirai1.0-small 미발견 | Weather benchmark 미발견 | 검증 불가 | I16 미발견 | source/model/data/env |
| B 압축+시간 잔차 | AdaPTS 미발견 | Moirai1.1-small 미발견 | Weather benchmark 미발견 | 검증 불가 | 호환 기준 미확인 | source/model/data/env |
| C 사후 scale 결합 | MSFT 미발견 | Moirai1.0-small 미발견 | Weather benchmark 미발견 | 검증 불가 | I16 및 scale predictions 미발견 | 공통 기준 예측 필요 |

MSFT expected pin은 `e848f23a2e3445df3f1c0ecd970d0a2e7ace3dec`, AdaPTS expected pin은 `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`이다. 이는 첨부 문서의 지정값이며 실제 로컬 checkout에서 검증한 commit이 아니다. 소스가 없어 q/k/v의 scale별 소유 구조·freeze_Wo·비-LoRA 모듈, AdaPTS linearAE의 미분 경계, MSFT sample 결합을 직접 검증하지 못했다. 문서의 설명을 로컬 구현 검증 결과로 바꾸지 않았다.

## 환경

| 환경 | Python | Torch 메타데이터 | Lightning / Uni2TS |
|---|---|---|---|
| 기존 `.venv` |3.11.15|2.8.0|모두 미설치|
| `.venv-channel` |3.11.15|2.7.1|모두 미설치|
| Anaconda base |3.13.9|2.5.1+cu121|모두 미설치|

기존 `.venv`의 Chronos-forecasting은2.3.2다. 설치된 배포판 메타데이터만 읽었으므로 import/실행 호환성을 보장하지 않는다. MSFT fork와 기존 Chronos 환경을 섞거나 변경하지 않았다.

GPU는 RTX3080 10,240MiB, driver580.178.04다. 마지막 메타데이터 확인 시 free8,956MiB였고 compute 목록은 RustDesk272MiB였다. 모델 로드나 메모리 적합성 검사를 하지 않았으므로 batch8 또는1024step/30분 가능성을 확인했다고 쓰지 않는다.

## 자료와 표본 지원

현재 저장소와 지정 sibling/cache 범위에서 공개 Weather benchmark를 찾지 못했다. 따라서 원 출처·hash·timestamp 열·수치 채널·행 수·간격, 공식 loader의 정확한70/10/20 정수 경계는 미확인이다. 기존 weather_runs/verification parquet와 다른 실험의 weather 예측 파일은 이름만 관련 있어 대체하지 않았으며 수치를 읽지 않았다.

N과 날짜가 없어 V_SELECT/CAL/PILOT_EVAL의 실제 index, H96 legal origins, unique 날짜, 유효 target 수도 계산하지 않았다. 상태는 **검증 불가**이며 `INSUFFICIENT_SUPPORT` 또는 0개 표본으로 표기하지 않는다. 뒤20% TEST의 수치 분석·통계 계산·채점을 하지 않았다.

탐색은 PRECHECK.json에 열거한 현재 저장소와 알려진 sibling7개 및 Hugging Face cache로 제한했다. source git remote는 제한된6개 root에서 깊이5까지8개 config를 확인했다. 전체 디스크에 없다는 주장은 아니다. TATO의 Moirai shell과 다른 외부 연구의 Moirai 언급은 있었지만, 요청된 source checkout·snapshot·Weather I16 checkpoint의 존재 증거는 아니다.

## 재사용과 남은 입력

C는 호환되는 I16이 없다면 A/C 공통 기준 학습1회와 scale별 예측 생성 비용이 필요하다. 이번에 그것을 실행하거나 C가 즉시 학습0회로 가능하다고 주장하지 않는다. 재사용에는 source pin·snapshot·채널/모드·L/H/patch/scales·split/scaler·seed·checkpoint 선택·sample-mean estimator 및 기존 TEST/VAL 노출 이력이 모두 맞아야 한다. 현재 이 조건들을 확인할 cache가 없다.

후속 실행 전에 필요한 것은 지정 pin의 원본 소스, 같은 공개 Weather 원본과 출처, 지정 모델 snapshot/config, 분리된 호환 환경, 이를 기준으로 작성·봉인된 고정 실행 코드다. 이번에 새 `.py`/쉘 실행기를 만들거나 원본 runner를 실행하지 않았다.

## 예산과 종료

문서상의 후속 제안은 A4+B6=10 jobs/최대10,240updates(FM 역전파8개, 작은 head2개), C의 CAL 볼록결합1회다. 실측 실행 예산 또는 현재 실행 권한으로 취급하지 않았다. 실제 model load0 / 학습0 / 추론0 / optimizer0 / 설치0 / 다운로드0 / TEST채점0 / CAL fit0이다.

**과학적 Go/No-Go 미평가.** 기술 입력 누락을 후보 실패로 판단하지 않는다. P0의 PRECHECK 두 파일을 기존 승인된 저장소에 게시하는 것으로 종료하고 구현·학습을 자동 시작하지 않는다.

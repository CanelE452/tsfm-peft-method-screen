# 세 후보 환경·자료 설치 완료 — 2026-09-25

상태: **INSTALLATION_VERIFIED / EXPERIMENT_NOT_STARTED**.
사용자의 “설치해줘” 승인에 따라 기존 P0의 설치·자료 확보 제한만 이 범위에서 해제했다. 실제 후보 runner 구현, 모델 로드·추론·학습은 수행하지 않았다. 기존 P0와 연구 결과를 덮어쓰지 않는다.

## 설치 결과

| 항목 | 확보·검사 결과 |
|---|---|
| MSFT | 지정 commit `e848f23a2e3445df3f1c0ecd970d0a2e7ace3dec`, 별도 환경 85 packages |
| AdaPTS | 지정 commit `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`, 별도 환경 130 packages |
| AdaPTS용 공식 Uni2TS | 공식 저장소 확보 시 HEAD `cfd46d4510ed8896f263116f32928eede05b0a75`, 버전 2.0.0으로 고정 |
| MSFT용 Uni2TS | MSFT fork 1.1.1 사용; AdaPTS 환경과 namespace 분리 확인 |
| 공통 환경 | Python 3.11.15, Torch 2.4.1+cu121, Lightning 2.4.0, NumPy 1.26.4 |
| Moirai-1.0-R-small | 공식 `Salesforce/moirai-1.0-R-small`, revision `f5bd5d01f0d67de856107d43abdd637380aae0a3` |
| Moirai-1.1-R-small | 공식 `Salesforce/moirai-1.1-R-small`, revision `0c24ab99db2c1a70ea2a0fc03bf113329772ac64` |
| Weather | 공식 `thuml/Time-Series-Library`, revision `2b66e59ee19dac8f6f19fb5d4997f289fdfea357`, `weather/weather.csv` |

두 가상환경에서 필요한 모델·adapter 클래스의 import와 `uv pip check`가 통과했다. 공식 Uni2TS의 `torch<2.5`, NumPy 1.26 의존성에 맞춰 호환 버전을 선택했다. 구버전 datasets의 PyArrow 호환성 등 설치 제약은 [constraints.txt](constraints.txt)에 기록했다. 이는 학습률·방법 정의의 변경이 아니다. 전체 설치 버전은 [MSFT freeze](msft.freeze.txt), [AdaPTS freeze](adapts.freeze.txt)에 고정했다. 설치 메타데이터의 Lightning 배포판 두 이름(`lightning`, `pytorch-lightning`)의 실제 버전도 freeze에 그대로 남겼다.

기존 `.venv` 및 `.venv-channel`의 전체 설치 배포판 이름·버전 목록은 설치 전과 동일하다. 기존 모델, 결과, 학습 환경을 교체하지 않았다. 검증 범위는 [설치 검증 JSON](INSTALL_VERIFICATION.json)과 [설치 전 목록](SETUP_BASELINE.json)에 있다.

## 자료 검사

공식 MSFT README → 공식 TSLib README → 저자 조직의 Hugging Face 배포 경로로 확보했다. [TSLib 공식 자료 안내](https://github.com/thuml/Time-Series-Library#prepare-data). 모델·CSV의 SHA256과 공식 LFS SHA256을 대조했다. 작은 config/README도 로컬 SHA256을 기록했다. [자산 manifest](ASSET_MANIFEST.json).

Weather는 52,696행·21채널이다. 원 MSFT `_load_custom`의 `int(N*.7)`, `int(N*.1)`, `int(N*.2)`를 그대로 적용한 행 경계는 다음과 같다. 우측 경계는 제외한다.

| 구간 | 행 범위 | 원점 수 L=2000 / L=512 | 원점의 날짜 수 |
|---|---|---:|---:|
| TRAIN | [0, 36887) | 34792 / 36280, stride 1 | 243 / 253 |
| V_SELECT | [36887, 38643) | 1661 / 1661, stride 1 | 12 |
| CAL | [38643, 40399) | 18 / 18, stride 96 | 12 |
| PILOT_EVAL | [40399, 42156) | 18 / 18, stride 96 | 12 |
| TEST 미사용 | [42156, 52695) | 계산·평가하지 않음 | — |

정수 내림 때문에 마지막 1행은 세 외부 분할 밖에 남는다. PILOT_EVAL은 최소 8개 날짜 조건을 만족한다. 이는 표본 개수 조건이며 실험 성공·충분한 검정력의 증거는 아니다. TRAIN/validation 42,156행에서 비유한 값은 0개였다. 전체 CSV의 날짜 열·hash·행 수만 전체 범위에서 읽었고, **TEST 수치 열은 읽거나 정규화·채점하지 않았다**.

원자료 TRAIN 날짜에는 2020-05-12 06:00 중복 1건과 2020-05-29 09:30→11:10의 100분 간격 1건이 있다. 원자료는 수정·보간·삭제하지 않았다. 따라서 위 원점 수는 **공개 benchmark의 행 순서 기준**이며 모든 창이 실제 시각상 균일 10분 간격이라는 인증이 아니다. 미래 runner에서도 이 차이를 숨기지 않아야 한다. 일부 채널명의 대체 문자도 원파일 그대로 보존했다. 날짜 파싱 첫 시도는 dayfirst 설정 때문에 실패했으나 실제 ISO 형식을 명시하여 재검사했고, 데이터나 경계는 바꾸지 않았다. [상세 메타데이터 검사](WEATHER_METADATA_AUDIT.json).

## 아직 남은 작업

설치·다운로드 차단은 해소했다. 다음 항목은 이번 설치 작업의 완료로 간주하지 않는다.

- 계약에 맞는 A/B/C 고정 실행 코드와 TEST 접근 차단 loader.
- 실제 모델 로드, A의 파라미터 공유·초기 동등성, B의 encoder까지의 gradient 전달, checkpoint 복원 검사.
- 기준 MSFT I16 학습 checkpoint와 scale별 예측 cache. 호환 checkpoint를 확보하거나 승인된 I16 학습이 필요하며, C의 준비 비용을 0으로 표시하면 안 된다.
- 본학습·선택·평가·자원 측정. 현재 import 검사는 CUDA 모델 실행이나 역전파 가능성을 보증하지 않는다.

**이번 실행량: 새 fits 0, optimizer updates 0, 모델 로드 0, 모델 예측 0, target error scoring 0.** 이전 지시문의 자동 학습 제한을 그대로 유지한다. 다른 MAG/ISO-NE 실험도 재개하지 않았다.

## 사용 및 재현

저장소 루트에서 MSFT는 `.cache/tsfm_three_candidate_setup_20260925/envs/msft/bin/python`, AdaPTS는 `.cache/tsfm_three_candidate_setup_20260925/envs/adapts/bin/python`을 사용한다. 활성화하지 않고 이 경로를 직접 지정하면 기존 Chronos 환경과 섞이지 않는다. source checkout과 모델/CSV의 정확한 로컬 상대 경로는 검증 JSON과 manifest에 있다.

[재설치 안내](REPRODUCE_SETUP.md). 두 모델 가중치는 합계 약 111 MB, Weather 약 7.2 MB다. 의존성·소스 포함 로컬 setup 폴더의 `du` 표시량은 약 9.2 GiB이며 uv cache와 hardlink가 있어 실제 추가 디스크 점유와 같지는 않다.

원자료·가중치·가상환경은 Git 제외 로컬 cache에 둔다. GitHub에는 버전, hash, 검사 결과와 문서만 공개한다. GitHub checkout만으로 즉시 수치 실험을 재생할 수 있다고 주장하지 않는다.

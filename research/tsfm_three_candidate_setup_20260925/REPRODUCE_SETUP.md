# 설치 재현 및 환경 경로

이 문서는 설치용이다. 모델 로드·추론·학습 runner가 아니다. 저장소 루트에서 실행한다. 기존 동일 경로가 있으면 새로 덮어쓰지 말고 먼저 manifest와 확인한다.

## 소스

| 로컬 경로 (`.cache/tsfm_three_candidate_setup_20260925/` 아래) | 공식 origin | commit |
|---|---|---|
| sources/MSFT | https://github.com/zqiao11/MSFT.git | e848f23a2e3445df3f1c0ecd970d0a2e7ace3dec |
| sources/AdaPTS | https://github.com/abenechehab/AdaPTS.git | 8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278 |
| sources/uni2ts | https://github.com/SalesforceAIResearch/uni2ts.git | cfd46d4510ed8896f263116f32928eede05b0a75 |

각 origin을 해당 경로로 clone하고 위 commit을 detached checkout한다. fork 및 공식 Uni2TS를 같은 환경에 설치하지 않는다.

## 환경

위 checkout을 먼저 갖춘 상태에서 Python 3.11.15와 uv를 사용한다. 기존 환경은 변경하지 않는다.

```bash
uv venv --python 3.11.15 .cache/tsfm_three_candidate_setup_20260925/envs/msft
uv pip install --python .cache/tsfm_three_candidate_setup_20260925/envs/msft/bin/python -r research/tsfm_three_candidate_setup_20260925/msft.freeze.txt
uv venv --python 3.11.15 .cache/tsfm_three_candidate_setup_20260925/envs/adapts
uv pip install --python .cache/tsfm_three_candidate_setup_20260925/envs/adapts/bin/python -r research/tsfm_three_candidate_setup_20260925/adapts.freeze.txt
```

초기 설치는 `constraints.txt`를 적용한 editable source 설치였다. freeze 파일은 실제 설치된 전체 버전과 상대 source 경로를 기록한다. 버전 목록은 wheel hash lock이 아니므로 재다운로드 wheel까지 동일함을 보장하지 않는다.

## 자료와 모델

`ASSET_MANIFEST.json`의 repo/revision을 지정한 Hugging Face `snapshot_download` 또는 `hf_hub_download`로 다운로드한다. 모델의 허용 파일은 `config.json`, `model.safetensors`, `README.md`; 데이터는 `weather/weather.csv` 하나다. 캐시는 `.cache/tsfm_three_candidate_setup_20260925/hf`이다. 다운로드 뒤 각 파일 SHA256을 manifest와 대조한다. 모델 `from_pretrained`나 forecast 메서드는 호출하지 않는다.

## 설치 확인

`uv pip check --python <각 환경의 python>`으로 의존성을 확인한다. import 확인은 `CUDA_VISIBLE_DEVICES=''`로 수행했다. MSFT에서는 `uni2ts.model.multi_scale_moirai`, AdaPTS에서는 `adapts.adapters`, `adapts.adapts`, `adapts.icl.moirai`와 `uni2ts.model.moirai`를 import한다. 객체 생성이나 forward는 포함하지 않는다. sandbox에서 uv cache 쓰기가 제한되면 `UV_CACHE_DIR=/tmp/tsfm_setup_uv_check`, Matplotlib 설정은 `MPLCONFIGDIR=/tmp/tsfm_setup_mpl`로 지정할 수 있다.

실제 import 경로와 의존성 검사 출력은 `INSTALL_VERIFICATION.json`에 있다. metadata 검사는 MSFT source의 외부 분할 정수 내림을 적용하고 `pd.read_csv(..., usecols=['date'])`로 전체 날짜만 읽었다. 수치 읽기는 `nrows=42156`으로 TRAIN/validation에 한정했다. origin은 `range(max(split_start,L),split_end-96+1,stride)`이며 CAL/PILOT stride=96이다. 원래 MSFT의 전체 CSV 정규화 loader나 AdaPTS full runner를 실행하지 않았다.

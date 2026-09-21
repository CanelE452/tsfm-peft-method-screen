# Rollout uncertainty PEFT CLI bundle

## 전달
`MASTER_CLI.txt` 하나가 최종 실행 계약이다. 나머지는 검증용 참조 부품이다.
기존 HIER/MAG 지시문과 합치지 않는다.

## 이번에 작성·검사한 것
- 2개 자료, 3개 학습군, 24 fits / 12,288 main updates + 12 smoke 계약.
- 첫 관측과 생성 관측의 구분, 생성 lead·예측 폭 저장, 4단계 causal rollout.
- STATE와 UNCERTAINTY의 동일 구조 8,744-parameter adapter.
- pinball / coverage / width, 초기화·gradient·보정 identity의 CPU 테스트 13개 통과.
- 참조 실행 환경: Python 3.13.5, torch 2.10.0+cpu. 이것은 사용자 GPU환경 검증이 아니다.

## 이번에 하지 않은 것
- 실제 데이터 다운로드/품질·split 확인.
- Chronos/LoRA/GPU 연결과 native branching parity.
- 새 학습, 실제 예측, 성능 채점, 사용자 저장소 쓰기.
- 새 제안의 성능 또는 신규성 검증.

`reference_core.py`는 NumPy 상태관리와 PyTorch 모듈의 작은 참조 구현이다.
실제 GPU 학습 driver는 CLI가 구현해야 한다. NumPy reference를 training loop에
직접 끼워넣어 모델 forward까지 no_grad로 만드는 식으로 사용하면 안 된다.

## 핵심 수정: 현재 공식 Chronos-Bolt는 중앙값-only가 아니다
2026-09-21에 읽은 공식 `chronos_bolt.py`의 `predict()`는 64시점 이후에
9개 분위수 경로를 확장하고 81개의 조건부 분위수를 축약한다.
따라서 F0의 약한 중앙값 rollout 하나를 이겼다고 새 방법 성공으로 평가하지 않는다.
F0_NATIVE, R_NATIVE, R_MC16, Chronos2 직접256 예측, 같은 CAL보정 기회를 포함한다.

## 원문과 공식 출처
1. Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated? (2026/ICLR)
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html
   https://arxiv.org/html/2510.16060v2
   https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration
   읽은 코드 revision: b60bfd92ff836525773c71039a83ba2fe3d123bf
2. Chronos official current implementation
   https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py
   읽은 file git blob SHA: db44821f4c657fa1da1219d8ff7f996594e375cc
   학습환경에서 실제 revision/file hash를 다시 봉인해야 한다.
3. Native configuration
   https://huggingface.co/amazon/chronos-bolt-small/blob/main/config.json
   https://huggingface.co/amazon/chronos-2/blob/main/config.json
4. Electricity author distribution
   https://github.com/laiguokun/multivariate-time-series-data
   https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz
5. ETT author distribution
   https://github.com/zhouhaoyi/ETDataset
   https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

## 실행 전 참조 테스트
```bash
python -m pytest -q test_reference.py
```
GPU모델을 받지 않고 실행된다. 실모델 smoke는 MASTER의 별도 항목을 따라야 한다.
실험 과학적 결과가 없는 현재 시점에 README를 실행완료 보고서로 사용하지 않는다.

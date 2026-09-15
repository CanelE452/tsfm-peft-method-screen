# 교정 표본 MOMENT 비교

[사용자 실행 계약](../results/channel_controlled_improvement_v1_20260916/PROTOCOL.md), [재사용 감사](../results/channel_controlled_improvement_v1_20260916/REUSE_RECEIPT.md), [실행 상태](../results/channel_controlled_improvement_v1_20260916/status.json).

독립 경로에서 36개 고정 trajectory를 실행한다. 8개 완전 기존 경로 재사용, 28개 신규. 두 LR·세 seed·20 epochs, corrected TRAIN, V_FIXED/V_MIXED 선택 분리. E_FIXED/E_MIXED는 기존 개발 기간이며 독립 test가 아니다. GPU lock/안전 감시와 4시간 상한, 추가 연구 자동 실행 없음.

명령: `bash scripts/with_cuda.sh .venv-channel/bin/python scripts/run_channel_controlled_improvement.py all`. 최초 prepare 뒤 smoke→train→select→evaluate→resources→verify/report를 연결한다. `status`는 읽기 전용. `resume`은 같은 fit의 완전 상태와 committed update journal을 확인하며 모호한 중단을 자동 재학습하지 않는다.

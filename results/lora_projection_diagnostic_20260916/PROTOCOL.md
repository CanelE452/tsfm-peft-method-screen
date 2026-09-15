# 저장 LoRA의 Q/K·V 구성요소 진단 — 신규 학습0

기준0855c83. 사용자 활성 목표의 실패 원인 검토에 따라 완료된 balanced LoRA4개 checkpoint를 재사용한다. 원래120-fit 건물 계약과 이후 완료된 실험은 다시 열지 않는다. 이 실행은 후보 학습이나 독립 성능 검증이 아니다.

## 질문·고정 비교

동일 용량 마지막 adapter보다 좋은 encoder LoRA는 어느 projection 변화에 의존하는가? 전력/교통×seed41000/41001의 기존 V 선택 checkpoint를 고정한다. 각 checkpoint의 학습된 head는 그대로 두고 다음 네 상태를 순서대로 검증한다.

- FULL: 학습된 q/k/v LoRA 모두 유지.
- QK_ONLY: v LoRA B만0으로 설정.
- V_ONLY: q/k LoRA B만0으로 설정.
- NONE: q/k/v LoRA B 모두0. **이는 따로 학습한 head-only가 아니다.**

A 및 head·backbone 가중치는 그대로 두며 매상태 원본 checkpoint에서 복구한다.8 layers×q/k/v의24개 B가 대상이다. 최대16개 V 예측, 신규 fits0/optimizer updates0. 데이터·모델·32채널·96→96·BF16 batch8·원점·정규화는 기존과 동일. V는 이미 선택과 분석에 사용된 개발 자료다. E 예측/점수는 이번 진단에서 읽거나 생성하지 않는다. 데이터 manifest 전체해시 검증에 E staged 파일해시는 포함될 수 있으며 E 내용의 분석과 구분한다.

FULL은 기존 저장 V 예측과 bitwise exact를 요구한다. 실패하면 허용오차를 올리거나 BF16 설정을 바꾸지 않는다. finite 출력, 선택된 B만 변경, head와 frozen parameters/buffers 불변, 각상태 후 원본 복구, 기존 결과와 checkpoint/data/source hash 보존을 확인한다. MSE/MAE는 독립 float64 scalar atol/rtol1e-12로 검산한다. 현재 승인된 RustDesk만 예외로 두는 전역GPU lock/guard를 사용하고, 시작30초/free4GiB, 실행중free1GiB, 누적대기600초/controller1800초를 지킨다. 재시도·추가구성·후속학습0.

## 해석

네 상태의 원점수와 전체 LoRA 제거 대비 감소량을 제시한다. QK 삭제 비용=L(V_ONLY)-L(FULL), V 삭제 비용=L(QK_ONLY)-L(FULL). 대칭 배분 QK=.5[(L(NONE)-L(QK_ONLY))+(L(V_ONLY)-L(FULL))], V는 나머지다. 두 배분 합은 L(NONE)-L(FULL)과 같다. 이는 이 checkpoint의 상호작용을 포함한 설명용 분해다. 비율 분모가0이면N/A, 음수면 방향 그대로 공개하며 필수 구성요소를 성급히 선언하지 않는다.

학습 후 제거는 공동 적응을 깨는 개입이다. QK만/V만 처음부터 학습한 결과, 최적 rank/삽입 위치, 새 메커니즘 성능을 대신하지 않는다. V의 변화도 뒤 layer의 QK 입력을 바꾸므로 QK=시간만/V=내용만이라는 엄격한 분리는 아니다. LoRA 자체가 알려진 방법이고 이 진단은 신규성 증거가 아니다. 새 PASS 기준을 만들지 않는다. 다음 가설에 필요한 증거를 얻되 시간 관계 가설을 결과에 맞춰 확정하지 않는다.

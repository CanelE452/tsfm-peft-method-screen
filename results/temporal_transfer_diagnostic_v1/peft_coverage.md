# PEFT 실행 이력 범위

LoRA 외에도 head/readout-only와 side network가 실제 학습·평가됐고, 내부 residual/conditional adapter도 LoRA와 결합해 실행됐다. H_FULL은 원래 출력층만 학습하는 선택적 파라미터 적응이다. 반면 bias/norm-only, soft prompt/prefix, 공식 IA3/ReFT, QLoRA는 이번 검사 범위에서 실행 근거를 찾지 못했다. 이것은 전체 과거의 부재 증명이 아니다.

peft_family_inventory.csv에 코드·결과·데이터·파라미터·상태와 별도 축을 나눴다. 내부 additive adapter를 Houlsby 재현으로, conditional LoRA gate를 IA3 재현으로, activation INT8을 QLoRA로 세지 않았다. Full-FT는 PEFT가 아니며 F0와 출력 혼합도 자체 PEFT 학습이 아니다. Covariate-trust의 Chronos adapter라는 이름은 모델 학습 어댑터가 아니라 호출 wrapper였고, 별도의 회귀 선택기는 TSFM PEFT로 세지 않았다.

검사 커밋: {"mltimeseries": "f490b67a99721f572f049e17382d5d431d15b2c2", "forecast-revision-peft": "237d2456991fbdabc764a4be1e4e18d2fbbd16fd", "covariate-trust-pilot": "2fe2443b2d24c09ad184387b7f7287f32e0f4cd6"}

참조 main의 tree와 28개 선택 파일을 읽었다. 모든 과거 branch/file을 전수 조사했다고 주장하지 않는다. Study35 원예측은 확인한 로컬 위치에 없어서 저장 점수/코드만 참조했고 alpha 곡선을 새로 계산하지 않았다. Study35 HEAD도 D에서 F0보다 악화됐으므로 전체 문제를 백본 표현 손상만으로 설명할 수 없다. JOINT에서 표현 손상이 기여했을 가능성은 미확정이다.

주 저장소 HEAD는 지시문의 3750490과 같고 새 완료 학습 결과는 없었다. 앞서 만든 미커밋 감지/정정 파일은 보존했고 이 작업은 commit/push/자동 재개를 하지 않았다.

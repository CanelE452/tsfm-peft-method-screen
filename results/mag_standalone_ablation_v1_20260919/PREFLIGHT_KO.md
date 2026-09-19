# 실행 전 감사

기준 HEAD: 8e51964d9910514c1104385ccb19d818901055e7

단일 계약 CONTRACT.txt를 보존했다. 기존 MAG_ONLY는 no-LoRA가 아닌 B0+MAG이다. B0 q/v rank8 LoRA294912와 residual8712를 구분한다. 새 F0 경로는 pretrained snapshot을 직접 로드하며 attach_lora를 호출하지 않는다. 실제 CPU모델18경로의 parameter/module검사, 동일seed초기값, native F0/step0/off예측정확동일을 확인했다. 기존24개모델·96개예측view와TRAIN/V/E hash검증. 기존결과수정0. CPU검사는 실제학습smoke의대체가 아니다.

새16fits/16384+8updates만승인. E는 모두기존노출개발패널이며NESO도독립source아님. GPU는실행시별도감시하고RustDesk만예외. 상세path/hash는BASELINE_MANIFEST.json. 원자료·가중치·prediction은로컬캐시다.

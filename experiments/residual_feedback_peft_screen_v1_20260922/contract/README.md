# Residual-feedback PEFT CLI bundle

- 두 후보 함께: `00_MASTER_CLI.txt`만 최종 계약으로 사용.
- A만: `01_A_AMORTIZED_CLI.txt`.
- B만: `02_B_PARTIAL_CLI.txt`.
- 별도 파일은 공통 계약의 같은 사본이다. 서로 다른 설계안을 합치지 않는다.
- `reference_core.py`는 모델 runner가 아닌 테스트 가능한 참조 부품이다.
- `test_reference_core.py`: 작은 CPU tensor에서 18개 검사. Chronos/데이터/GPU는 미검사.
- `TEST_RESULTS.txt`에 실제 실행 결과를 보존했다.

새로운 연구 가설이므로 실제 성능이나 신규성을 확보했다고 주장하지 않는다.
기존 저장소를 읽었지만 이 번들을 작성하는 동안 사용자 저장소를 수정하거나 GPU 연구를 실행하지 않았다.

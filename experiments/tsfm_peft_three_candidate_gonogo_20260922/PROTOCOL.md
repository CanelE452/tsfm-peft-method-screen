# 실행 전 고정한 세부 규칙

최상위 계약은 contract/MASTER_PLAN.txt이며 USER_EXECUTION_REQUEST.txt는 추가 실행 지시이다. 기존 실험의 학습 산출물을 가져오지 않는다. 새 경로 이름은 예시와 다를 뿐 독립 범위는 동일하다.

- Python 3.11 기존 환경은 읽기 전용 재사용한다. bitsandbytes 0.50.2만 이번 cache의 python_packages에 설치했다. GPU BF16/NF4 실측 통과. torch CUDA12.8, RTX4070.
- seed92201/92202 모두 반복이다. LoRA alpha=rank, scaling=1, dropout0. 학습 원점마다 같은 batch8; 평가 batch64는 메모리 검사 후 고정한다. optimizer AdamW eps1e-8, betas(.9,.999), weight_decay0, clip1, scheduler 없음.
- 분위수는 학습 손실과 채점에서 오름차순 정렬한다. 원래 crossing은 별도 보존한다. native64 원단위 예측만 사용한다. sigma는 최초50% population std(ddof0).
- checkpoint 동률이면 앞선 checkpoint. strongest baseline은 seed별 자기 V로 고정한다. 정확히 같은 점수면 계약의 arm 나열 순서. 후보는 baseline 선택에 포함하지 않는다.
- Q double quant=true, compute BF16. backend 기본 고정밀 예외 유지. 모든102 eligible linears에 LoRA; 그룹 비용에는 고정밀 linear의 LoRA도 포함한다. Q_IO16은 입력/출력 ResidualBlock 내 모든 quantized linear를 BF16으로 유지한다.
- QERA-diag는 BF16 pretrained model의 같은32 TRAIN contexts에서 RMS input activation을 FP32 제곱/FP64 누적으로 수집한다. zero RMS만 1e-8로 대체한다. 공식 대각 S 및 SVD 수식과 수치 대조한다. quantization residual의 원본은 safetensors 원본으로 LoftQ와 동일하다. 전체 QERA 논문 재현이라고 부르지 않는다.
- Q 민감도는 각 example의 sigma로 나눈 BF16 대 NF4 차이 제곱의 전체 평균 감소이다. 미래 target 접근 없음. 음수0, 모두0이면 uniform4. rank를 모든 eligible group에1부터 배정, 상한16, 고정 그룹 순서 ties. 미사용 비율>=5%는 mismatch/HOLD. 같은 init family의 초기화 통계와 원본 weight 접근 기회 동일.
- Q artifact는 실제 직렬화된 base packed/scales/exceptions/config 파일 + 선택 adapter 파일 전체 bytes. BF16 참조 분모는 adapter 없는 BF16 base 파일. 각 방법의 전체 저장 크기로 low-bit baseline 자격을 고정한다.
- T 학생 프로세스에는 별도 BRIDGE-only 배열과 global row offset만 전달한다. OLD raw 파일/전체 data.load()를 학생 학습 경로에서 호출하지 않는다. scale와 teacher예측 허용. 교사 cache는 학생의 고정 TRAIN tuple union 및 전체 BRIDGE_VAL contexts에 생성한다. seed별 A1 개별 사용. 학생 pseudo/true gradient norm 각각 기록, lambda0.5 고정.
- F 공유 optimizer는 매 round/client마다 초기화, private optimizer state는 client마다 유지한다. LOCAL optimizer도 매4updates 경계 초기화. A는 seed별 동일 고정, B만 집계. private와 shared를 한 optimizer.step에 포함해 local update를 중복 계산하지 않는다. 서버는 B payload와 client scalar V만 받는다.
- F origin o는 첫 target의 zero-based row이며 h=0..63; phase o+h와 slope h/64를 사용한다. 실제 달력 아님. client별 객체와 데이터 접근 분리.
- 모든 optimizer.step은 호출 전 장부에 예약하며 실패/중단도 소모로 남긴다. main8192, smoke24 초과 금지. main 시작 전 후보별 source seal. 재실행은 완료 checkpoint/장부 확인 없이는 금지.
- candidate와 각seed의 V-selected baseline이 모두 checkpoint0>128>256(FL0>8>16)로 엄격 감소하면 최적화 여지 flag와 HOLD를 우선한다. 비학습 N0 baseline에는 적용하지 않는다. 기준선 충분의 보조 해석은 Q에서 Q_FP 대비1% 이내, T에서 N0 또는 RECENT 대비 후보 추가이득 미확보, F에서는 LOCAL과 SHARED 대비 단순 개인화의 양성 효과를 명시한다. GO_STANDARD_ONLY가 후보의 연구 GO를 뜻하지 않는다.
- paired noncircular moving block bootstrap: TEST6h origin56개(14일), 2000회, RNG92200, 모든 series와 두seed를 함께 보존. seed별 및 점수평균 CI 별도. CI로 자동 판정하지 않는다.
- 모든 선택 봉인과 TEST prediction 저장을 완료한 후에만 TEST 채점한다. 추가 실험/자동 v2 없음. 각 후보 BLOCKED는 과학적 판정과 분리한다.

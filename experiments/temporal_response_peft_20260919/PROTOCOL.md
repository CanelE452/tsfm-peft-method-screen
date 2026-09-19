# 시간 반응 보존 PEFT — 고정 방법 파일럿

2026-09-19. 사용자의 새 방법론 연구 요청에 따라 작성. 이전 C3 계약을 수정하거나 실패 판정을 덮어쓰지 않는다. 본 문서는 새 후보 하나의 개발 파일럿이며 논문 성공 보증이 아니다.

## 문제와 방법

잘 학습된 B0 위에 작은 어댑터를 더하면 입력 오류에 적응하는 과정에서 B0의 지속 변화 반응도 바뀔 수 있다. 기존 gate는 입력 위치의 추가 보정만 제한하며 출력 반응 보존을 직접 최적화하지 않는다. 기존 진단은 이 가설의 동기이며, 이 손실이 최종 성능을 개선한다는 증거는 아니다.

동결된 B0를 f0, 추가 어댑터가 있는 모델을 fθ, 관측 과거 x의 최근 구간에 수준 변화를 더한 입력을 Tx라고 한다. 후보 TRP(Temporal Response Preservation)는

`L = L_2pinball(fθ(x), y) + mean(abs((fθ(Tx)-fθ(x))-(f0(Tx)-f0(x)))/sigma)`

를 최적화한다. 출력 9 quantiles × 64 horizons 전체에 적용하고 teacher는 stop-gradient다. 추가 보정 rθ=fθ−f0의 변화 `rθ(Tx)−rθ(x)`를 억제한다. 일반 증류처럼 보정 자체를 0으로 밀지 않는다. 이는 유한 변화 응답 매칭이며 미소 Jacobian 일치의 새 이론이라고 주장하지 않는다. 훈련 후 추론은 기존 C2와 동일한 8,712-parameter adapter 한 개만 사용한다. teacher 이중 추론이나 시나리오 식별은 배포에 필요 없다.

## 비교군과 정보 권한

| 군 | 추가 손실 | 해석 |
|---|---|---|
| PLAIN | 0 | 기존 C2; 동일 경로 4개 재사용 |
| ANCHOR | 두 입력에서 mean(abs(fθ−f0))/sigma의 평균 | 단순 보정 크기 억제로 충분한가 |
| SHUFFLE | TRP와 같되 시간 위치를 섞은 변화 | perturbation 값 분포·L2 크기가 같아도 시간 구조가 필요한가 |
| IDEAL | mean(abs(fθ(Tx)−fθ(x)−a))/sigma | B0 반응 대신 이상적 영구 수준 이동 a를 강제하면 충분한가 |
| TRP | 위 후보 손실 | B0의 시간 구조 반응 보존 |

B0, 과거 관측 x, TRAIN-only sigma만 입력 권한이다. y는 기존 supervised loss에만 사용한다. 원래 clean x, generator state, fault mask, 진짜 synthetic delta, 미래 y를 변환·teacher·model input으로 사용하지 않는다. 여기서 a는 기존 사건의 숨긴 delta가 아니라 관측 x 위에 별도로 생성하는 새 probe 값이다. IDEAL은 모든 미래에도 a가 지속된다는 추가 귀납 가정을 대조한다. TRP는 그런 미래 정답을 만들지 않는다.

probe: epoch/기존 sample index/source로 키를 정한 seed91901, 길이 {16,32,64,128} 균등, 부호 ±1 균등, 크기 U[2,8]×TRAIN sigma. suffix와 그 512자리 permutation의 값 histogram은 정확히 같다. probe draws는 모든 arm/seed에서 같다. 평가 상태 이름·오류 위치는 model에 전달하지 않는다. 변환된 입력의 새 supervised labels는 없다. SHUFFLE의 위치가 시간 전체에 퍼지는 점은 의도된 대조이며 완벽한 난이도 동일성 보증은 아니다.

## 고정 예산과 재사용

- Electricity·ETTm1, 기존 TRAIN/V_SELECT/E_DISCOVERY origin·채널·분할을 그대로 사용한다. 전력16계열 전이는 기존 E이며 독립 자료가 아니다. 모든 E가 반복 노출된 개발자료다.
- Chronos-Bolt-small FP32, TF32 off, 기존 B0 seed81551/81552 동결. rank8 adapter, 동일 초기값/배치 구성·순서. 추가 trainable8,712, B0 LoRA294,912는 이미 학습되었고 이번에는 동결.
- 각 source × seed에서 5군. PLAIN 4개는 receipt·가중치·학습 데이터·일정 검증 후 재사용. 신규 4군 × 2source × 2seed = 최대16fits, 16,384 main updates. smoke 최대16updates. 새 LR/seed/candidate 자동 추가 없음.
- LR3e-4는 두 source의 기존 C2 V-only 선택값. 이번 후보 결과를 보기 전에 고정. λ=1, AdamW(.9,.999), eps1e-8, weight_decay0, clipnorm1, batch32, microbatch32, 32epochs ×32updates=1024. 메모리 부족 시 배치 크기나 수치 설정을 몰래 바꾸지 않고 자원 차단.
- checkpoint0/256/512/768/1024. 기존 5상태 REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8 동등 가중 V nMAE로 선택, 동률이면 이른 step. fixed1024도 모두 보고. 추가 LR 선택·출력 보정 없음.
- 네 신규 군 모두 1update마다 32개 supervised labels, adapted forward2회 및 frozen teacher forward2회. PLAIN보다 훈련 계산비용이 높고 정확도와 비용을 별도 보고한다. 동일 optimizer 기회이지 동일 FLOPs라고 쓰지 않는다. 기존 C3/MAG/B0는 맥락 비교이며 이번 동일 손실 대조와 구분한다.
- GPU 독점 lock, 외부 compute는 기존 승인 RustDesk만 예외. 여유 VRAM1GiB, RAM2GiB, disk10GiB 유지. 불안전하면 wait/정확한 차단 기록. fit별 실패로 독립 나머지 군을 자동 취소하지 않되 공통 코드/데이터 문제면 중단한다.
- optimizer intent와 완료 journal, 매 epoch 복원점. 미완료 epoch를 임의 재학습하지 않는다. 실패한 update도 예산에서 지우지 않는다.

## 평가·검산·판단

전체 학습과 선택 파일 봉인 후 Electricity/전력16계열/ETTm1의 standard10상태와 shape9상태, selected/fixed1024 예측을 모두 먼저 저장한다. 그 뒤 E labels를 채점한다. 원점수 nMAE·MAE·pinball, seed별, 오류 FAULT/REFERENCE/SHIFT4/SHIFT8/SHIFT_POINT 및 모든 shape 손해를 남긴다. 주 비교는 TRP 대 PLAIN/ANCHOR/SHUFFLE/IDEAL이며 기존 MAG/C3/B0도 선택·LR 차이를 표시해서 보고한다.

효과 크기와 주 단위 paired bootstrap 2,000회(두 seed를 함께 평균)를 보고하되 이는 seed 모집단/탐색 이력의 불확실성이 아니다. 주 family: 전력 전이 SHIFT8의 위 네 비교에 Bonferroni4 95% 구간. 사전 주제 투자 신호는 네 비교에서 각각 평균 nMAE 1% 이상 감소 및 해당 구간 하한>0, 두 seed에서 모두 같은 개선 방향, 전력 전이 REFERENCE·FAULT 손해가 각1% 이내일 때에만 기록한다. 이는 운영상 다음 검토 기준이며 논문 PASS 기준이 아니다. 모든 ETT 반대 결과를 유지하며 여기서 실패해도 모든 robust PEFT를 반증하지 않는다. 미충족은 NO_METHOD_EVIDENCE_AT_FIXED_PROTOCOL 또는 UNCERTAIN이며 결과에 맞춰 문턱을 바꾸지 않는다.

CPU 부품 검사 외에 실모델 초기 B0 동일성·두 업데이트·유한 gradient·고정 가중치/버퍼 불변·checkpoint 복원·adapter off B0 동일성 검사. 모든 condition의 첫/끝 origin과 첫/끝 channel에서 독립 scalar MAE/pinball 대조. 데이터·코드·checkpoint SHA-256 봉인. 보고서는 실행 완료, 예측 효과, 시간 반응 추가 가치, 비용, 신규성을 분리한다.

## 신규성 범위와 선행

- [Time-PEFT 실제 run.py](https://github.com/kaist-dmlab/TimePEFT/blob/main/run.py): FFT 주파수 top-k 처리와 채널별 bottleneck, MOMENT encoder LoRA·head 학습. 그 구현을 복제하거나 정식 비교했다고 주장하지 않는다.
- [Srinivas & Fleuret, ICML2018](https://proceedings.mlr.press/v80/srinivas18a.html): Jacobian matching/노이즈 증류·전이 관계. 반응 보존 원리 자체는 알려져 있다.
- [Geometric Alignment Distillation, 2026 preprint](https://arxiv.org/abs/2606.01651): 다른 도메인에서 교사-학생 입력 변화 반응을 맞추는 가까운 아이디어. 정식 게재 여부는 확인하지 않았다.
- [δ-Adapter 공식 코드](https://github.com/Anoise/Adapter): 입력/출력 추가 보정과 중요한 입력을 보호하는 mask. 작은 추가 모듈이라는 사실만으로 신규성을 주장하지 않는다.

현 단계의 주장 후보는 '시계열 robust 추가 PEFT에서 teacher의 유한 지속 변화 반응을 보존하는 학습 규칙'이다. 일반 regularization보다 좋아야 하고, 기존 matching을 시계열에 적용한 정도라는 반론도 남는다. 좋은 파일럿이라도 독립 source, 아직 노출되지 않은 확인 자료, 강한 정식 PEFT 비교, 추가 seed 및 신규성 검토 전에는 방법론 논문 완성/게재 가능을 선언하지 않는다. 이 파일럿 이후 추가 학습은 자동 연결하지 않는다.

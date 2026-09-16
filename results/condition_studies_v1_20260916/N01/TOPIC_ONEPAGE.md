# N01 ASYNC


5. N01 ASYNC — 타깃의 최신값은 늦고 다른 채널은 최신일 때
======================================================================

데이터: Electricity, TRAIN에서 정한4채널, C336h,H48h.
문제: 타깃의 미관측 suffix를 아직 관측된 donor 채널 정보로 보완할 수 있는가?
선행 단서: t-PatchGNN(ICML2024)의 비동기 채널 관계. 이번 간단한 ridge/imputer는 그 논문 재현이 아님.
기존 Freshness와 차이: 모든 채널을 함께 지연시키지 않고 한 타깃만 지연.

관측 조건:
TRAIN 원점마다 타깃 c=(origin_index mod4), 지연 d={0,6,24}를 epoch별 순환.
타깃의 [o-d,o)값만 감춘다. 다른3채널은 o까지 관측.
V/E에서는 모든 타깃을 순회하고 d=0/6/24를 공통 평가.
추가 E 진단: d=12(학습 안 한 길이), 모든 채널 d=24(도움될 최신 donor 부재).
주지표는 부분 지연 d=6/24를 동일 가중한 타깃 normalized RMSE.
정상d0 손해와 d12/all-delayed는 별도 표. 실제 지연 빈도라고 부르지 않는다.
loss는 해당 원점의 지정 타깃만 사용. 모든 군은 동일한 true training target을 받음.

허용 정보: 가려지지 않은 값, 관측 mask M, 각 위치의 마지막 관측에서 경과한 시간 A.
가린 target context 값으로 auxiliary reconstruction supervision을 새로 주지 않는다.
TRAIN 완전관측 과거로 imputer 회귀식을 fitting하는 권한은 모든 군에 동일하게 허용.
따라서 ridge fitting에는 기본 예측 학습과 다른 within-TRAIN 회귀 쌍이 추가된다.
그 회귀 쌍 수를 별도로 세며 A1 대 A0를 순수한 구조 효과라고 부르지 않는다.
핵심 A3/A2 비교는 같은 회귀식·같은 회귀 학습 정보를 공유한다.
보조 mask/age는 모든 군에 past-covariate 행으로 제공. native target NaN과 구분.
NaN을 숫자0으로 바꾸면서 실제 관측값이라고 속이지 않는다.

고정 단순 회귀 imputer:
z값 TRAIN에서 채널c를 다른3채널의 동시점 값+상수로 ridge 예측.
ridge alpha={.1,1,10}, TRAIN 내부3개 forward-chaining fold로 선택. E/V를 사용하지 않는다.
마지막 관측 유지값 l_c(t), ridge r_c(t)를 계산.
donor도 없는 진단 상태는 donor LOCF와 mask 사용; 전체 회귀량을 관측값으로 취급하지 않음.

4개 학습군:
A0 NATIVE: target suffix NaN을 native mask로 처리 + 공통 M/A + LoRA.
A1 RIDGE: 결측 위치만 r_c(t)로 채움 + M/A + LoRA.
A2 RESID: r_c(t)+theta_c^T phi(t)로 채움 + M/A + LoRA.
   phi=[1,l_c,r_c, donor_z*M(3), donor_M(3)] (총9), theta 초기0.
A3 AGE_RESID: A2와 같되 + eta_c * a_c(t)*(r_c(t)-l_c(t)).
   a=min(A/24,2), eta초기0. 관측 위치는 항상 원래 값 그대로.
   이 추가항은 [미검증] 작은 입력 보정이며 '새 age-aware 기법'이라고 전제하지 않는다.
A2/A3 보정은 missing positions만 적용, 실제 원입력 mask는 유지해 model이 fill을 구분.
계산값은 표준화 공간에서 만들고 raw 단위로 복원해 native 입력에 넣는다.

직접 대비: A3 vs A2 (age 상호작용), A3 vs A1 (학습 가능한 보정의 필요성), A3 vs A0.
A3 추가 계수4개도 파라미터 표에 포함. A2보다 큰 용량일 수 있음을 숨기지 않음.
기대: 부분 지연의 예측 개선, all-delayed에서는 장점 약화 가능.
반례: A1이나 native grouping만으로 충분하면 별도 imputer PEFT 필요성 약함.
상관이 강한 채널만 E에서 골라 결과를 살리지 않음.

필수검사: 숨긴 suffix poison 입력불변, donor 변화에는 입력반응, 미래donor 사용0,
관측값 불변, 네 target의 노출균형, A2/A3 초기=A1, eta=0 축소 동작.
한 wrapper 초기결과가 native F0와 다른 것은 imputation 차이일 수 있으므로
초기 identity는 '같은 전처리의 LoRA0'와 비교한다.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.

# R04 REVISION


8. R04 REVISION — 정확도를 유지하며 불필요한 예측 수정 줄이기
======================================================================

데이터: Traffic 4채널, C336h/H48h, paired origins o와o+24.
같은 미래를 예측한 구간은 early[24:48]와 late[0:24]. 다른 lead를 그대로 빼지 않는다.
TRAIN/V/E의 두 horizon이 각 role 경계 안에 있어야 한다.
두 입력은 서로 다른 forward 또는 서로 다른 group ID. 뒤 문맥이 앞 예측에 새어 들어가면 안 된다.

[확인] 옛 Candidate04도 paired origin과 F0 대비 correction revision을 다뤘다.
이번 차이: 고정 point 목적과 새로운 공통 분할, innovation 정보의 작은 직접 대조.
새 데이터 독립 재현이나 원 논문 전체 재현이라고 부르지 않는다.
선행: Using dynamic loss weighting to boost improvements in forecast stability.
예측 안정성 손실/동적 가중 자체는 알려진 접근이다.

기본 L_task = 두 origin의 채널별 normalized MSE 평균.
R_raw = mean[((late[0:24]-early[24:48])/sigma)^2].
Delta = 현재 예측 - 같은 입력의 고정 F0 점예측.
R_corr = mean[((Delta_late[0:24]-Delta_early[24:48])/sigma)^2].
F0 예측은 원점별 캐시 가능. 미래 정답 없는 no-grad 호출만 사용.

새로 관측된 정보:
v_c(o)=mean[((observed_y[o:o+24]-F0_early[0:24])/sigma_c)^2].
이는 late 원점에서는 이미 관측된 값이며 late미래정답이 아니다.
w_c(o)=1/(1+v_c(o)). TRAIN 원점 전부에서 구한 mean(w)로 나눠 평균1.
V/E에는 TRAIN mean을 고정 사용. v를 미래 정답/최종 forecast error로 정의하지 않는다.

단일 lambda:
TRAIN의 첫16pair에서 F0의 L_task와 R_raw를 기록.
lambda=clip(.05*median(L_task)/max(median(R_raw),1e-8),1e-4,1).
초기 R_corr=0을 분모로 쓰지 않는다. 모든 규제군에 같은lambda.
값/gradient 비율은 다를 수 있으므로 큰 clipping 빈도도 보고. 사후lambda재튜닝 금지.

4개 학습군:
D0 PLAIN: L_task.
D1 STABLE: L_task+lambda*R_raw (기존 안정성 손실의 제한된 대조).
D2 CORR: L_task+lambda*R_corr (옛 FR원리의 새 데이터 비교).
D3 INNOV: L_task+lambda*mean[w_c*((Delta_late-Delta_early)/sigma)^2].
D3는 '새 정보가 크게 들어왔으면 보정수정을 덜 억제'하는 [미검증] 가중 규칙.
coefficient가 작은 것만으로 좋아지지 않도록 w의 TRAIN 평균을1로 맞췄다.

CPU 단순 대조: 선택된D0 late 예측과 이미 발행한early 겹침 예측의 blend.
alpha={0,.25,.5,.75,1}, V에서 선택, 같은 실제 미래만 blend.
발행한 과거 예측 파일은 나중 모델로 덮어쓰지 않는다.

선택/목표:
D0은 V accuracy최저. 다른 군은 각 checkpoint의 V accuracy가
선택한D0 V normalizedRMSE의1.01배 이하인 후보 중 revision RMS최저를 선택.
조건을 만족하는 checkpoint가 없으면 그 군의 accuracy최저와 INFEASIBLE를 기록.
LR 후보 모두에게 같은 rule. 1%는 이번 안정성 목적의 설계허용치이며 논문합격선 아님.
주 결과: accuracy vs raw revision의 paired frontier.
E에서도 accuracy보호를 만족하는지 별도 확인. 수정량만 줄어든 것을 성공이라 하지 않음.
'불필요한' 수정의 진짜 레이블은 없으므로 accuracy제약하 revision이라는 조작적 정의.
큰 혁신 구간에서 반응이 느려지는지 TRAIN 기준v 상위quartile을 진단하되 E로 cutoff를 정하지 않음.

필수검사: exact target timestamps overlap, late정보->early전달0,
innovation값의 as-of 가용성, F0 disabledLoRA복원, lambda초기분모유한,
두forward 비용을 한forward효율이라고 세지 않음.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.

NO_GO_NO_HEADROOM

대상: M5 일 단위 (d_1914–d_1941, 적격 30,458계열 전수). RAF 미측정.
근거: G2-B 실패 — 목표 데이터로 학습한 LightGBM이 zero-shot Chronos-2를 RMSSE에서
      0.50%만 이긴다(기준 1.0%). 학습 원점 4→12개, 공변량 없음→포함 세 설정 모두 미달.
      지시문 §6 "B 실패 → NO_GO_NO_HEADROOM (A 결과와 무관), G3 진행하지 않는다".

게이트별 관찰
  G1(i)   ZS-QMEAN PRB = -1.65%   CI[-2.38,-0.92]  상한<0            PASS
  G1(ii)  ZS-QMEAN fill 0.8664 vs SBA 0.9456, 차 -0.0792
          CI[-0.0820,-0.0766]  상한<0                                PASS
  G1(iii) (intermittent+lumpy) - smooth = -0.0282 CI 상한 -0.0108    PASS
          추출교란: ZS-MED 상한<0 이나 ZS-QMEAN CI 0 미포함 -> 해당없음
  G2-A    ZS-best=ZS-Q90 Stock@90 16.381 -> NZ-QMEAN 9.209
          ΔStock% +43.78%  CI[+41.13,+46.54]  (기준 3.0)             PASS
  G2-B    ZS-QMEAN×α* RMSSE 0.7547 -> REF-LGB×α* 0.7509
          ΔRMSSE% +0.50%  CI[+0.38,+0.63]  (기준 1.0)                FAIL
  G3      미실행 (G2-B 실패)

설정: L=7, R=7, τ=0.9, context 512, h=28, α격자 0.50–4.00(0.05), 부트스트랩 2000회 seed 20260922.
평가기: 논문 run_raf_instancenorm.py 정의 이식 (점예측 clip0, σ̂=IQR/1.35, order-up-to).
NZ: 같은 파일 patch_instancenorm 이식, chronos 2.3.2 inner_model.instance_norm 에 적용 확인.

해석 경계
  - G0(논문 재현) 미실행. 측정 도구 타당성은 코드 이식 수준에서만 담보된다.
  - RAF 미측정. 지시문은 "대상 데이터셋 모두"를 요구하나 M5 실패만으로 G2-B는 실패한다.
  - 단일 test 원점(d_1913). 계열 부트스트랩은 창 공통요인을 독립으로 센다.
  - Stock@90은 지시문이 새로 정의한 지표다(논문 평가기는 재고를 출력하지 않는다).

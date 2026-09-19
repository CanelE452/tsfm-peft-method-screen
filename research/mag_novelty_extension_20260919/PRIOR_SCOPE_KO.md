# 학습 중 확인한 추가 선행과 주장 범위

2026-09-19. 현재 PETSA 8회 대조의 방법·표본·기준을 변경하지 않는 문헌 검토다. 여기서 새 비교군을 추가하거나 학습 결과를 선행해 판정하지 않는다.

[AIRA, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_AIRA_Activation-Informed_Low-Rank_Adaptation_for_Large_Models_ICCV_2025_paper.html)는 activation outlier 정보를 LoRA 초기화·rank 배분·학습에 활용한다. 따라서 ‘진폭 또는 outlier 정보를 PEFT에 이용한다’는 넓은 주장은 MAG의 신규성이 될 수 없다. AIRA의 activation 기반 weight 적응과 MAG의 raw 관측 robust 통계에 의한 추가 patch residual 제한은 구체적인 위치와 계산이 다르다. 이번 확인은 공식 초록·논문 검색 본문 범위이며 공식 전체 구현 재현이나 성능 비교가 아니다.

[Han·Qu, arXiv:2608.30502v1](https://arxiv.org/html/2608.30502v1)는 동결 TSFM 위 Kalman adapter의 온라인 update를 제한하는 gate와 robust update 대조를 다룬다. 논문은 다음 update 전에 전체96-step label을 공개하는 immediate-reveal replay를 명시하고, update label에 오염을 주며 깨끗한 label로 채점한다. MAG는 E 전에 가중치를 고정하고 관측 입력만으로 내부 잔차를 제한한다. 서로 다른 정답 도착·오염 위치·적응 절차를 같은 실험으로 간주할 수 없다. 그 차이와 별개로, ‘동결 TSFM 위의 보정 제한’ 자체도 독창적인 일반 원리라고 주장하면 안 된다. 이 선행의 runtime와 수치를 재현한 것은 아니다.

현재 남길 수 있는 차이는 **이미 학습된 B0와 원래 관측을 유지하면서, 고정 robust 진폭 gate를 attention 이전의 작은 추가 잔차에 적용하는 구체적인 설계**다. 기존 직접 대조의 이득은 이 설계에 대한 경험적 근거다. 충분한 방법론 신규성은 검색 결과의 부재나 좁은 구조 차이만으로 증명되지 않는다. AIRA·온라인 Kalman 논문의 결과를 MAG와 숫자로 합치지 않으며, 별도 학습 대조가 완료됐다고 표시하지 않는다.

공식 일차 자료만 주장 근거로 사용했다. 검색에 나타난 요약 사이트는 근거로 사용하지 않았다. 공식 CVF PDF 직접 열기는403으로 실패했으며 공식 초록 페이지와 검색에 노출된 논문 텍스트를 확인했다. arXiv 논문은 본문 §1/§4/AppendixB를 확인했다. 검색을 이유로 현재 봉인한 PETSA 실험을 중단하거나 교체하지 않는다.

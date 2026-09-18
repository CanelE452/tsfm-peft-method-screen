# 비교군과 선행의 정보 권한

| 대조 | 입력·학습 | 이번 역할 | 정식 선행 재현 여부 |
|---|---|---|---|
| C0/B0 | 이미 source TRAIN에서 학습한LoRA; E에서동결 | 추가이득의출발점 | 기존고정방법재사용 |
| C1/C2/C3/MEAN/ROTATE16/RECENCY | 동일source TRAIN/V;기존가중치 | 추가계산·용량·지속성·보정량·위치·최근성 | 기존계약의완료비교재사용 |
| POS_ONLY | observed값과무관한32위치gate+8712adapter | 학습된위치사전정보 | 새설명대조,8744params |
| MAG_ONLY | observed크기/robust scale,연속성없음 | 크기만으로충분한지 | 새설명대조,8712params |
| OUTPUT_CONTEXT | B0median64+observed구간평균8,sourceTRAIN학습/E동결 | 저비용outputresidual | COSA에서동기를얻은offline구조대조,4673params |
| F0 | pretrained고정;같은observed입력 | 추가적응전기준 | 기존예측재사용 |

[Houlsby/ICML2019](https://proceedings.mlr.press/v97/houlsby19a.html)의 작은 어댑터와 [UniPELT/ACL2022](https://aclanthology.org/2022.acl-long.433/)의 학습 gate 원리는 기존이다. UniPELT는 여러PEFT 모듈을 결합하는 NLP framework이며 본 위치gate 대조를 그 정식 재현이라고 부르지 않는다.

[TAFAS/AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/33965)와 [COSA/ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html)는 온라인 적응/도착정답의 권한을 갖는 관련 선행이다. OUTPUT_CONTEXT는 source TRAIN만 학습하고 평가정답으로 갱신하지 않으므로 원논문의 전체코드·온라인절차 재현이 아니다. [TATO/ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48c5226582f41254026748c7e35d4ac2-Abstract-Conference.html)의 변환탐색 전체도 이번에 실행하지 않는다. 기존 공식코드 감사의 원문·hash는 부모 결과의 official_source_receipts.json/LITERATURE_BOUNDARY.md로 보존하며 새 UniPELT/TAFAS의 공식게재 페이지를 이번에 확인했다.

같은 학습 기회는 완전한 capacity/최적화 난이도 일치를 뜻하지 않는다. OUTPUT_CONTEXT는 분위수 간격을 보존하는 제한된 보정이며 C3는 간격도 바꿀 수 있다. 서로 다른 정답 권한의 온라인 성능과 이번 점수를 단순 순위화하지 않는다.

# 선행 경계 — 2026-09-15 재확인

- [확인] [Time-PEFT 공식 코드](https://github.com/kaist-dmlab/TimePEFT)와 [ICML 2026 목록](https://icml.cc/Downloads/2026)은 temporal/multichannel complexity에 따른 frequency/channel adapter 연구를 제시한다. 이번 단변량·달력 관측 개수에 따른 출력 혼합은 그 구현을 재현하는 실험이 아니다. ICML poster61767 원문 페이지는 웹 도구에서 fetch 실패했으므로 전체 논문을 읽었다고 하지 않는다.
- [확인] [Park 등 논문 원문](https://www.merl.com/publications/docs/TR2026-030.pdf)은 Chronos의 건물 데이터 제한 이력 적응과 full FT/LoRA를 비교한다(§2, §4). 따라서 건물에 LoRA를 쓴다는 것은 신규성이 아니다. [저자 기관 서지](https://www.merl.com/publications/TR2026-030)는 DOI2025·December2025와 BibTeX March2026을 함께 표시하므로 연도 차이를 보존한다. Article116446, volume348이다.
- [확인] [BuildingsBench 원문/서지](https://arxiv.org/abs/2307.00142)와 [공식 loader](https://github.com/NatLabRockies/BuildingsBench)는 건물 단위 zero-shot/transfer benchmark를 제공한다. 본 파일럿의 3/14일 계약은 공식 6개월 transfer benchmark와 다르다.
- [확인] [In-Context Fine-Tuning, ICML2025](https://proceedings.mlr.press/v267/faw25b.html)은 관련 시계열 예시를 문맥에 제공하도록 학습한 TimesFM 적응을 다룬다. 관련 예시 활용과 few-shot 자체를 새 기여로 주장하지 않는다.
- [확인] [CFTR 출판사 초록](https://www.sciencedirect.com/science/article/abs/pii/S0378778826005906)은 unseen-building few-shot transfer와 hour/season residual correction·similarity weighting을 명시한다. Article117530, volume363, 2026-07-15. 출판사 검색 색인의 초록을 확인했으며 유료 전문은 확보하지 못했다. 상세 구현까지 비교했다고 주장하지 않는다.

[확인] 검색식 `forecast-state coverage LoRA`, `weekday LoRA shrinkage`, `coverage-aware fine-tuning forecast` 및 위 다섯 최소 선행에서 이 파일럿의 c/(c+tau) 건물 규칙과 정확히 같은 직접 선행은 확인하지 못했다. 검색 실패는 선행 부재의 증명이 아니다.

[추정] q0+alpha(qLoRA-q0)는 일반적인 예측 혼합/수축이며 c/(c+tau)는 표본 수에 따른 신뢰도 수축 형태다. 이 식만으로 새 PEFT 학습 방법의 신규성을 주장하기 어렵다. NOVELTY_STATUS=KNOWN_RULE_FORM_UNRESOLVED_APPLICATION. 직접적인 건물 weekday/weekend PEFT 동치 여부는 미확정이다.

[설계] 이 파일럿은 cross-building source transfer, residual correction의 새로운 구조, in-context 예시 공급을 추가하지 않는다. Stage A 상호작용과 단순 대조군 대비 추가 가치를 먼저 검사한다. 좋은 개발 점수가 나오더라도 논문 신규성 또는 확증 PASS로 바꾸지 않는다.

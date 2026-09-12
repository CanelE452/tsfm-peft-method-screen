# Candidate05: repaired Round1 — FAIL

[문제·방법]
부분적으로 도착한 정답으로 업데이트하면서 미공개 horizon의 업데이트 직전 예측을 보존하는 Maturity-PEFT. [고정 실험 계약](../../docs/CANDIDATE_05_RECOVERY.md).

[데이터·학습 파라미터]
Jena, 336 context / 48 horizon, 24시간 간격 30 origins, seed30000, LR1e-4, lambda1, eligible origin당 8 updates. 수정 전과 데이터·설정·평가 기준 모두 동일. TAFAS-like는 GCM만 재현하며 fixed scheduling/native probabilistic loss를 사용한 제한적 baseline이다.

[실행·원본 보존]
실행 commit `886dc58f718fa79fba8f24b3fd337c60e3addcec`. 사용자의 “나머지해줘” 요청에 따라 5개 스트림을 추가 실행했다. [기존 중단 결과](../candidate_05/RESULT.md)는 그대로 보존했다. 기존에 관측된 개발 E를 재사용한 구현 복구 실험이며 새로운 holdout 검증은 아니다. E 관측 후 설정 조정은 하지 않았다.

[raw 결과]
| Method | Scaled 2-pinball ↓ | Unrevealed drift ↓ | Worst 5 loss ↓ | Adaptation seconds |
| --- | ---: | ---: | ---: | ---: |
| F0 | 0.583857107 | 0.000000000 | 0.985529176 | 0.000 |
| IMMEDIATE_LORA | 0.646770200 | 0.386186084 | 1.240334796 | 22.622 |
| WAIT_FULL | 0.662482790 | 0.000000000 | 1.250890626 | 21.653 |
| TAFAS_LIKE | 0.630833796 | 0.272694681 | 1.083491480 | 11.828 |
| MATURITY_PEFT | 0.655929350 | 0.380202982 | 1.243728896 | 24.090 |

[강한 단순 baseline·relative 결과]
최강 online baseline: TAFAS_LIKE. Proposed gain = 100 × (baseline loss − proposed loss) / F0 loss = **-4.298236% F0** (필요 ≥1%). F0 대비 gain은 -12.344158%다.

[무결성]
5개 스트림 모두 30회 발행 완료. 150개 예측 파일과 최종 평가 캐시 일치. 정답은 당시 index < now인 구간만 업데이트에 사용했다. 초기 예측 및 checkpoint replay 오차 0. 독립 지표 replay 최대 오차 1.11e-16. 기존 완료 F0/Immediate/WaitFull 예측 배열과 모두 정확히 일치. Preservation이 실제 업데이트에서 활성화됐다. 검증 상세: recovery_verification.json.

[성공/실패 판정]
**FAIL**. 최강 online baseline 대비 gain -4.298236% (≥1%), adaptation overhead 103.676% (≤20%), 마지막 5 origins 제외 gain 5.085003% (>0). 모든 조건을 만족해야 PASS다. 시간 수치는 이번 동일 GPU 순차 실행의 실측값이다.

[계산량]
추가 standard fits 0, streaming runs 5, optimizer updates 920, candidate wall 95.506s, peak allocated GPU 707079680 bytes. 누적 34 fits + 9 stream attempts (8 complete, 1 original abort). 기존 5-stream 예산 안에 든다고 주장하지 않는다.

[말할 수 없는 것]
단일 데이터·seed의 개발 결과이며 통계적 유의성, 다중 데이터 일반화, 신규성 확정 또는 미관측 holdout 성능을 주장할 수 없다. 겹치는 horizon이 있다. TAFAS의 전체 논문 재현 결과가 아니다.

[Round2 추천 여부]
추천하지 않는다. Round2는 실행하지 않았다.

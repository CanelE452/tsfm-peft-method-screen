# BOPTEST 신호 계약 — bestest_hydronic_heat_pump

2026-09-24. 출처 `ibpsa/project1-boptest` `0f8a467`, version 1.0.0-dev.
FMU(`wrapped.fmu`, sha `674b9500c1c89fdb`) 의 `modelDescription.xml` 과 `testcase.py` 를 읽어 만들었다.
시뮬레이션 실행 0회, BOPTEST 서비스 호출 0회. 공개 서비스는 여전히 해석 불가이며 이 자료는 소스에서 직접 얻었다.

## 1. 제어 명령 — 발행 / 유효 / 실행값의 3중 구분

`NEXT_STAGE_CONTRACT.md` §2 의 첫 칸("제어 명령의 의미·발행/유효/실행값 구분")이 여기서 채워진다.
BOPTEST 는 신호 하나를 세 이름으로 나눠 둔다.

```
<name>_u          우리가 발행하는 값        (input)
<name>_activate   그 발행을 유효화할지 0/1   (input)
<name>_y          실제로 적용된 값          (output)
```

`_activate = 0` 이면 `_u` 는 무시되고 내장 baseline 제어기가 돌며, 그때도 `_y` 는 실행값을 돌려준다.
따라서 "지시했지만 적용되지 않은 구간" 이 데이터에 그대로 남는다 — 과도응답 연구에서 반드시 구분해야 할 부분이다.

| 명령 | 단위 | 범위 | 뜻 |
|---|---|---|---|
| `oveHeaPumY_u` | 1 | 0 ~ 1 | 히트펌프 압축기 속도 변조 (0 정지, 1 최대) |
| `oveFan_u` | 1 | 0 ~ 1 | 증발기 팬 on/off |
| `ovePum_u` | 1 | 0 ~ 1 | 방열 회로 펌프 on/off |
| `oveTSet_u` | K | 278.15 ~ 308.15 | 실내 작용온도 설정값 |

전환 사건의 후보는 `oveHeaPumY` 의 0↔양수, `oveFan`·`ovePum` 의 0↔1, `oveTSet` 의 계단 변화다.

## 2. 목표 측정값 후보

| 신호 | 단위 | 뜻 |
|---|---|---|
| `reaTZon_y` | K | 실내 작용온도 |
| `reaTSup_y` / `reaTRet_y` | K | 바닥복사 공급/환수 수온 |
| `reaPHeaPum_y` | W | 히트펌프 전력 |
| `reaQHeaPumCon_y` / `reaQHeaPumEva_y` | W | 응축기/증발기 열량 |
| `reaCOP_y` | 1 | 성능계수 |

수온(`reaTSup_y`)이 명령 변화에 가장 빠르게 반응하고 실내온도(`reaTZon_y`)가 가장 느리다.
시상수가 다른 신호가 한 시스템 안에 같이 있으므로, 어느 것을 목표로 두느냐가 difficulty 정의를 바꾼다.
이것은 아직 정하지 않았다.

외생 입력으로 기상 관측 22종(`weaSta_rea*`)이 출력에 함께 나온다. 예보는 별도 `/forecast` API 다.

## 3. 시각 계약

```
시각 원점    1월 1일 00:00 = 0 초. 모든 시각은 그 해의 초 단위다
제어 주기    config 기본 3600 초. /step 으로 변경 가능하며 하한 검증은 "음수 금지" 뿐이다
결과 해상도  제어 주기가 30초 이상이면 내부적으로 30초 간격으로 저장된다
             (testcase.py: ncp = (end-start)/30, y_store 에 중간점 전부 누적)
시나리오 날짜 peak_heat_day = 23, typical_heat_day = 115 (연중 일 번호)
```

제어 주기가 1시간이어도 결과는 30초 간격으로 남는다. 과도응답이 데이터에 보인다는 뜻이며,
"1시간 간격이라 전환 응답을 못 본다" 는 걱정은 해당하지 않는다.

## 4. 아직 못 채운 칸 — 실행이 필요하다

| NEXT_STAGE_CONTRACT §2 항목 | 상태 |
|---|---|
| 제어 명령 의미·발행/유효/실행값 구분 | 채움 (위 §1) |
| target measurement 의미·단위·시각 | 후보와 단위는 채움. 무엇을 target 으로 둘지는 미정 |
| 정상 운전과 과도응답의 물리적 정의, TRAIN-only tau0 추정 | 미정 — 실제 응답 파형을 봐야 한다 |
| 전환 수, 구간별 날짜 수 | 미정 — 시뮬레이션을 돌려야 센다 |
| H·context·간격과 tau0 의 비율 | 미정 — tau0 이 없다 |
| 시간순 분할, 정답 겹침 방지 | 설계 가능하나 전환 수를 알아야 한다 |
| 평가 원점·날짜 가중 | 미정 |
| 최소 실용 차이 | 미정 — 센서 해상도·업무 허용오차 없음 |

## 5. 남은 한 가지 — Docker

```
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER      # 이후 재로그인 또는 newgrp docker
```

그 다음은 저장소에서 `docker compose up web worker provision` → `http://127.0.0.1:8000`.
`.env` 는 저장소에 기본값이 들어 있어 따로 만들 필요가 없다.
`web`/`worker`/`provision` 은 ubuntu 20.04 기반으로 로컬 빌드되고, `minio`·`redis` 는 pull 한다.
현재 여유 디스크 98G.

더 가벼운 길도 있다. 웹 API 없이 `worker` 이미지 안에서 `testcase.py` 의 `TestCase` 를 직접 몰아
데이터만 뽑으면 minio·redis·web 이 필요 없다. 어느 쪽이든 Docker 설치 이후의 일이다.

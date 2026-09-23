# BOPTEST 실행경로 감사 — 로컬과 public 을 분리

2026-09-23. HEAD `db3879c`. 조회만 수행했고 시뮬레이션 배포·advance 는 하지 않았다.

## 0. 판정

```
AUTH_OR_POLICY_BLOCKED
```

public web service 는 **문서상 존재하고 설치 없이 쓸 수 있다.** 그러나 이 실행 환경에서
도메인 자체가 DNS 해석되지 않아 도달하지 못했다. 서비스가 죽었다는 확인은 아니다.

## 1. 두 경로를 분리한다 (이전 A0 의 오류 정정)

| | 로컬 Docker 경로 | public web-service 경로 |
|---|---|---|
| 요구사항 | Docker 필요 | **"There are no installation requirements."** |
| 이 머신 상태 | docker 명령 없음, boptest python 모듈 없음 | 문서상 이용 가능 |
| 이번 확인 | 설치 금지라 미확인 | **DNS 해석 실패로 도달 불가** |
| 버전 | — | User Guide v0.9.0 기준 |
| 인증 | — | 문서에 인증·API 키 언급 없음 |
| 정책 | — | 비활동 시 자동 정지(로컬 기본 15분과 동일) 외 rate limit 명시 없음 |

**이전 A0 의 "docker 가 없으므로 시뮬레이션 경로도 막혔다" 는 틀렸다.** 로컬 경로만 보고
내린 결론이었고, public 경로를 확인하지 않았다.

## 2. 도달 시도 기록 (실측)

```
GET https://api.boptest.net/testcases
  curl      HTTP 000, 0 B, 0.257 s
  상세      * Could not resolve host: api.boptest.net
  WebFetch  getaddrinfo ETIMEOUT api.boptest.net

DNS 해석 대조
  api.boptest.net   해석 실패
  boptest.net       해석 실패      <- 도메인 전체가 안 잡힌다
  ibpsa.github.io   해석 OK
  github.com        해석 OK

같은 환경의 다른 https 호스트
  https://github.com                         HTTP 200
  https://ibpsa.github.io/project1-boptest/  HTTP 200
  https://pypi.org                           HTTP 200

resolver  nameserver 127.0.0.53 (systemd-resolved), search tail30f706.ts.net
proxy     환경변수 없음
```

두 경로(curl·WebFetch)가 모두 **DNS 단계**에서 실패했고 다른 호스트는 정상이다.
이 환경의 네트워크 정책일 가능성이 높으나, 공개 DNS 직접 조회 도구(dig/nslookup)가 없어
서비스 자체의 생존은 확인하지 못했다. 지시문대로 우회하지 않고 멈춘다.

## 3. 그래서 확인하지 못한 것

아래는 전부 **미확인**이며, 문서에 적혀 있다는 이유로 계약값으로 승격하지 않는다.

- testcase 목록에 `bestest_hydronic_heat_pump` 가 실제로 있는지
- `available inputs` 의 정확한 signal name / unit / description / range
- `measurements` 의 zone temperature / control state / 외기 변수 이름
- simulation step 의 시간 단위와 반환 timestamp 의 의미
- `POST /advance` 가 u 를 적용하는 구간과 반환 y 의 시각 (off-by-one)
- `PUT /forecast` 가 주는 값과 실제 미래 truth 의 구분
- hidden/internal state 로서 모델 입력에 쓰면 안 되는 값의 목록
- 제어 가능한 물리량이 zone setpoint 인지, heat-pump modulation 인지, supply setpoint 인지

**smoke simulation 은 수행하지 않았다.** 호출 0회, 배포 0회.

## 4. 문서에서 확인한 API 계약 (참고용, 로컬 대조 전)

`GET /version` `/name` `/measurements` `/inputs` `/step` `/scenario` `/forecast_points` `/kpi` `/submit`,
`PUT /step` `/initialize` `/scenario` `/forecast` `/results`, `POST /advance`.
배포된 테스트케이스는 base url 앞에 붙이고 요청에 `testid` 를 덧붙인다.
**testcase 목록 조회·배포 엔드포인트는 이 문서 범위에서 확인되지 않았다.**

## 5. 해제 조건

다음 중 하나가 충족되면 `PUBLIC_METADATA_READY` 로 올릴 수 있다.

1. 이 환경에서 `api.boptest.net` DNS 가 해석되도록 네트워크 정책이 열린다.
2. 사용자가 다른 환경에서 `GET /testcases` 응답을 받아 전달한다.
3. Docker 설치가 승인되어 로컬 경로가 열린다(이번 지시문 금지 항목).

## 6. 출처

- Getting Started — BOPTEST User Guide v0.9.0 (public web service, 설치 요구사항 없음, `https://api.boptest.net`)
  https://ibpsa.github.io/project1-boptest/docs-userguide/getting_started.html
- API Summary — BOPTEST User Guide v0.9.0 (엔드포인트 표)
  https://ibpsa.github.io/project1-boptest/docs-userguide/api.html

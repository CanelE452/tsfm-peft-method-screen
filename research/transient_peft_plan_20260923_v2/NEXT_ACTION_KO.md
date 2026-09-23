# 다음 한 단계

2026-09-23. HEAD `db3879c`.

## 판정

```
NEED_PUBLIC_BOPTEST_ACCESS
```

이전 A0 의 `STOP_DATA` 에서 바뀌었다. 데이터 경로가 없다고 본 것이 틀렸기 때문이다.
BOPTEST public web-service 는 설치 없이 쓸 수 있고, 막힌 것은 **이 실행 환경의 네트워크**다.

`STOP_DUPLICATE` 로 가지 않는 이유: 선행을 분리해 보니 verdict 가 서로 다르고, 이번 후보의
고유 질문(§2 의 2×2)이 이 저장소에서 아직 분리된 적이 없다.

## 막은 것과 푸는 법

```
막힌 것   api.boptest.net / boptest.net 이 이 환경에서 DNS 해석되지 않는다
          curl: Could not resolve host  |  WebFetch: getaddrinfo ETIMEOUT
          같은 환경에서 github.com, ibpsa.github.io, pypi.org 는 HTTP 200
서비스    생존 여부 미확인 (공개 DNS 직접 조회 도구 없음)
```

셋 중 하나면 풀린다.

1. 이 환경에서 해당 도메인 DNS 가 열린다
2. 다른 환경에서 `GET https://api.boptest.net/testcases` 응답을 받아 전달해 주신다
3. Docker 설치가 승인되어 로컬 경로가 열린다 (이번 지시문 금지 항목이라 별도 승인 필요)

2번이 가장 가볍다. 응답만 있으면 testcase 목록과 `bestest_hydronic_heat_pump` 존재 여부를
확인할 수 있고, 이어서 `/inputs`·`/measurements`·`/step` 메타데이터까지 같은 방식으로 받을 수 있다.

## 그 다음에 하는 것

접근이 열리면 순서대로.

```
1  testcase 목록 · 버전 · inputs / measurements / 단위 / 시각 계약 조회      (GET 만)
2  짧은 smoke 1회 — initialize 시각, 명령 적용 interval, advance 반환 y 의 시각,
   off-by-one, default control 과 명시 control 의 차이만 본다                (데이터셋 생성 아님)
3  G0 DATA SEMANTICS 판정
4  G1 tau0 추정 (TRAIN-only)
5  G2 history signal — SYS0~SYS3
6  G3 F0 / L0 / FI 만 학습하고 멈춤
```

**6 까지 가기 전에는 LM(제안) 을 학습하지 않는다.** FI 만으로 difficulty 가 해결되면 거기서 끝난다.

## 결정이 필요한 것 — 2개

**질문 1.** BOPTEST 접근을 어떻게 풀지. 위 1·2·3 중 하나.

**질문 2.** 이번 후보의 고유 질문이 "상태 생성이 물리 시간 상수를 가진 명시적 감쇠 동역학"이라는
한 겹으로 좁혀졌다. STAR 가 이미 `ΔW = A R B ⊙ D` 형태로 patch 별 조건부 저랭크 변조를 하고 있어,
변조 형태 자체의 신규성은 주장할 수 없다. 이 한 겹으로 진행할지.

## 이번 작업 실행 내역

```
PEFT fit               0회
모델 학습·추론          0회
BOPTEST 배포·advance   0회 (도달 실패, 호출은 GET 시도 2회뿐)
패키지·Docker 설치      0회
기존 파일 삭제·수정      0건
새로 만든 것            research/transient_peft_plan_20260923_v2/ 문서 6개
```

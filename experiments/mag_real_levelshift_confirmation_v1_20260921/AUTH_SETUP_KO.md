# ISO-NE 로컬 인증

사용자가 ISO Express 가입을 완료했다고 알렸다. 인증은 아직 실행하지 않았고 데이터 차단 상태도 아직 해소되지 않았다.

사용자의 일반 터미널에서:

```bash
cd /home/minjae/Documents/github/tsfm-peft-method-screen
.venv/bin/python experiments/mag_real_levelshift_confirmation_v1_20260921/setup_isone_auth.py
```

가입 이메일과 비밀번호를 프롬프트에 입력한다. 비밀번호를 채팅이나 명령 인수에 넣지 않는다. 공식 HTTPS API info의 JSON 응답을 확인한 뒤에만 `.cache/mag_real_levelshift_confirmation_v1_20260921/private_auth/isone.json`에 저장한다. 폴더0700/파일0600이며 Git ignored다. 로컬 파일에는 인증 정보 원문이 있으므로 공유하지 않는다. 리다이렉트는 따라가지 않고 인증 실패 시 저장하지 않는다. 스크립트는 학습을 시작하지 않는다.

정상 메시지는 `ISO_NE_AUTH_READY`다.401/403이면 이메일 인증과 공식 웹 로그인 가능 여부를 확인한다.

[공식 API 인증 문서](https://webservices.iso-ne.com/docs/v1.1/)는 HTTPS Basic Authentication을 요구한다. [공식 보고서 매핑](https://www.iso-ne.com/static-assets/documents/2017/06/webservices_documentation.xlsx)은 `/hourlysysload/day/{day}/location/{locationId}`와 control-area location32 및7년 보관을 안내한다. `/realtimehourlydemand`는 Hub/zone load obligation으로 별도 설명되어 있으므로 계약 자료와 혼동하지 않는다. API로 받은 데이터가 계약의 Hourly Real-Time System Demand와 의미·시간·단위가 같은지는 인증 후 추가 검증해야 한다. 이 문서는 source 계약 변경이나 자동 학습 승인이 아니다.

검사: 구문 검사, 비대화형 입력 거부, 인증 파일 경로 Git 제외 확인 완료. 실제 사용자 인증은 미실행이다.

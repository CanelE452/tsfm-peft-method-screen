"""Run in the user's own terminal. Never enter credentials through chat."""
import getpass
import json
import os
from pathlib import Path
import sys
import tempfile
import requests

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / '.cache' / 'mag_real_levelshift_confirmation_v1_20260921' / 'private_auth'
ENDPOINT = 'https://webservices.iso-ne.com/api/v1.1/hourlysysload/info.json'


def main():
    if not sys.stdin.isatty():
        raise SystemExit('일반 로컬 터미널에서 직접 실행하세요. 채팅에 인증 정보를 입력하지 마세요.')
    username = input('ISO Express 가입 이메일: ').strip()
    password = getpass.getpass('ISO Express 비밀번호 (화면에 표시되지 않음): ')
    if not username or not password:
        raise SystemExit('빈 인증 정보: 저장하지 않았습니다.')
    try:
        # Only the official HTTPS host receives authentication. No redirect following.
        response = requests.get(ENDPOINT, auth=(username, password),
                                headers={'Accept': 'application/json'},
                                timeout=40, allow_redirects=False)
    except requests.RequestException:
        raise SystemExit('공식 API 연결 실패. 인증 정보는 저장하지 않았습니다.')
    if response.status_code != 200:
        raise SystemExit(f'공식 API HTTP {response.status_code}. 인증 정보는 저장하지 않았습니다. 이메일 인증 및 웹 로그인 가능 여부를 확인하세요.')
    try:
        payload = response.json()
        assert isinstance(payload, (dict, list)) and payload
    except (ValueError, AssertionError):
        raise SystemExit('정상 API JSON이 아니므로 인증 정보를 저장하지 않았습니다.')
    DIRECTORY.mkdir(parents=True, exist_ok=True, mode=0o700)
    if DIRECTORY.is_symlink():
        raise SystemExit('인증 폴더가 symbolic link이므로 중단합니다.')
    DIRECTORY.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix='.credentials-', dir=DIRECTORY)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump({'username': username, 'password': password}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, DIRECTORY / 'isone.json')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print('ISO_NE_AUTH_READY — 공식 API 인증 성공. 인증 정보는 로컬 전용 파일에 저장했습니다.')
    print('학습 및 원자료 다운로드는 아직 시작하지 않았습니다.')


if __name__ == '__main__':
    main()

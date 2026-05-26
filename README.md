# Overwatch AI Coach MVP

Google Drive에 올린 수동 녹화 영상을 선택해서 Gemini 3.5 Flash로 코칭 분석하고,
결과를 Streamlit에서 보여준 뒤 JSON으로 다시 Google Drive에 자동 저장하는 로컬 MVP입니다.

## 포함 기능

- Google Drive 입력 폴더의 영상 목록 조회
- 역할(`tank`/`dps`/`support`) + 영웅 선택
- 후보 이벤트 구간 기반 분석
- `잘한 장면` / `보완 장면` 분리 피드백
- 공통 60점 + 포지션 특화 40점 구조
- 지표별 confidence 및 낮은 신뢰도 이유 표기
- 결과 JSON 자동 저장 및 `저장된 분석 보기`
- 한국시간 주간 분석 횟수 표시

## 설치

```bash
cd ow-coach-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit secrets

`.streamlit/secrets.toml`에 아래 값을 넣으세요.

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
DRIVE_INPUT_FOLDER_ID = "YOUR_INPUT_FOLDER_ID_OR_FOLDER_URL"
DRIVE_OUTPUT_FOLDER_ID = "YOUR_OUTPUT_FOLDER_ID_OR_FOLDER_URL"

GOOGLE_OAUTH_CLIENT_JSON = '''
{
  "installed": {
    "client_id": "YOUR_GOOGLE_OAUTH_CLIENT_ID",
    "project_id": "YOUR_GOOGLE_CLOUD_PROJECT_ID",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_GOOGLE_OAUTH_CLIENT_SECRET",
    "redirect_uris": [
      "http://localhost"
    ]
  }
}
'''

GOOGLE_OAUTH_TOKEN_JSON = '''
{
  "token": "YOUR_ACCESS_TOKEN",
  "refresh_token": "YOUR_REFRESH_TOKEN",
  "token_uri": "https://oauth2.googleapis.com/token",
  "client_id": "YOUR_GOOGLE_OAUTH_CLIENT_ID",
  "client_secret": "YOUR_GOOGLE_OAUTH_CLIENT_SECRET",
  "scopes": [
    "https://www.googleapis.com/auth/drive"
  ]
}
'''
```

## Online deploy

온라인 배포에서는 `localhost` OAuth 로그인 버튼을 그대로 쓰기 어렵습니다.
대신 로컬에서 한 번 로그인해서 생성된 `.streamlit/google_oauth_token.json` 내용을
배포 플랫폼의 시크릿 `GOOGLE_OAUTH_TOKEN_JSON`으로 넣어 주세요.

권장 순서:

1. 로컬에서 앱 실행 후 Google Drive 로그인 연결 완료
2. `.streamlit/google_oauth_token.json` 파일 내용 복사
3. GitHub에는 `secrets.toml`, `google_oauth_token.json`을 올리지 않음
4. 배포 플랫폼 시크릿에 아래 값 등록
   - `GEMINI_API_KEY`
   - `DRIVE_INPUT_FOLDER_ID`
   - `DRIVE_OUTPUT_FOLDER_ID`
   - `GOOGLE_OAUTH_CLIENT_JSON`
   - `GOOGLE_OAUTH_TOKEN_JSON`

이렇게 하면 배포 서버에서는 별도 `localhost` 로그인 없이
저장된 Google Drive 권한으로 바로 동작합니다.

## Google OAuth 설정

1. Google Cloud Console에서 `OAuth client ID`를 생성합니다.
2. 애플리케이션 유형은 `Desktop app`을 권장합니다.
3. 발급된 JSON 내용을 `GOOGLE_OAUTH_CLIENT_JSON`에 그대로 넣습니다.
4. 앱 첫 실행 후 `Google Drive 로그인 연결` 버튼을 눌러 본인 계정으로 승인합니다.

## Drive 동작 방식

- 입력 폴더: 로그인한 사용자의 권한으로 영상 목록을 읽습니다.
- 출력 폴더: 로그인한 사용자의 권한으로 분석 JSON을 자동 저장합니다.
- 저장된 분석 보기: 출력 폴더의 JSON을 다시 읽어 복기 화면을 재구성합니다.

## 로컬 토큰 파일

앱은 로그인 후 아래 파일에 OAuth 토큰을 저장합니다.

- `.streamlit/google_oauth_token.json`

이 파일은 `.gitignore`에 포함되어 있으며 커밋하면 안 됩니다.

## 주의

- 이 버전은 로컬 우선 MVP입니다.
- OCR/Optical Flow 고도화 전이라 일부 지표는 confidence가 낮을 수 있습니다.
- 공개 저장소로 올리기 전에는 실제 시크릿과 토큰 파일이 포함되지 않았는지 반드시 확인하세요.

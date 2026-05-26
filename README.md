# Overwatch AI Coach MVP

Google Drive에 올린 오버워치 녹화 영상을 선택해 Gemini로 분석하고, 결과를 Streamlit 대시보드와 JSON으로 저장하는 프로젝트입니다.

## Features

- Google Drive 입력 폴더의 영상 목록 조회
- 역할(`tank`, `dps`, `support`)과 영웅 선택
- 이벤트 후보 구간 기반 장면 분석
- 강점 / 약점 장면 분리
- 데스 원인 및 개선 과제 제시
- 적 조합 추정 기반의 더 구체적인 피드백
- 결과 JSON을 Google Drive 출력 폴더에 저장
- 저장된 분석 결과 다시 불러오기

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required Secrets

`.streamlit/secrets.toml` 또는 배포 플랫폼의 Secrets에 아래 값을 넣어야 합니다.

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

## Online Deploy

온라인 배포에서는 `localhost` 기반 OAuth 로그인 버튼을 그대로 쓰기 어렵습니다.
대신 로컬에서 한 번 로그인해서 생성된 `.streamlit/google_oauth_token.json` 내용을
배포 플랫폼의 `GOOGLE_OAUTH_TOKEN_JSON` 시크릿으로 넣어 사용하면 됩니다.

### Recommended Steps

1. 로컬에서 앱 실행 후 Google Drive 로그인 연결 완료
2. `.streamlit/google_oauth_token.json` 파일 내용 복사
3. GitHub에는 실제 `secrets.toml`과 `google_oauth_token.json`을 올리지 않기
4. 배포 플랫폼에 아래 Secrets 등록

- `GEMINI_API_KEY`
- `DRIVE_INPUT_FOLDER_ID`
- `DRIVE_OUTPUT_FOLDER_ID`
- `GOOGLE_OAUTH_CLIENT_JSON`
- `GOOGLE_OAUTH_TOKEN_JSON`

## Notes

- 이 프로젝트는 공용 Google Drive 권한을 사용하는 운영 방식에 맞춰져 있습니다.
- API 비용은 `GEMINI_API_KEY` 소유자에게 청구됩니다.
- 실제 시크릿 파일과 OAuth 토큰 파일은 저장소에 커밋하지 않아야 합니다.

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.genai import Client
from google.genai import types
from google_auth_oauthlib.flow import InstalledAppFlow

MODEL_NAME = "gemini-3.5-flash"
MODEL_LABEL = "Gemini 3.5 Flash"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = Path(__file__).resolve().parent / ".streamlit" / "google_oauth_token.json"
CATEGORY_LABELS = {"aim": "에임", "move": "무빙", "judge": "판단", "op": "운영"}
LANE_LABELS = {"strength": "잘한 장면", "weakness": "보완 장면"}
ROLE_WEIGHTS: dict[str, dict[str, int]] = {
    "tank": {"aim": 10, "move": 20, "judge": 30, "op": 40},
    "dps": {"aim": 40, "move": 20, "judge": 25, "op": 15},
    "support": {"aim": 15, "move": 20, "judge": 35, "op": 30},
}
ROLE_HEROES: dict[str, list[str]] = {
    "tank": ["D.Va", "Doomfist", "Junker Queen", "Mauga", "Orisa", "Ramattra", "Reinhardt", "Roadhog", "Sigma", "Winston", "Wrecking Ball", "Zarya"],
    "dps": ["Ashe", "Bastion", "Cassidy", "Echo", "Genji", "Hanzo", "Junkrat", "Mei", "Pharah", "Reaper", "Sojourn", "Soldier: 76", "Sombra", "Symmetra", "Torbjorn", "Tracer", "Venture", "Widowmaker"],
    "support": ["Ana", "Baptiste", "Brigitte", "Illari", "Juno", "Kiriko", "Lifeweaver", "Lucio", "Mercy", "Moira", "Zenyatta"],
}
TIMELINE_COLORS = {"engagement": "#4db6ff", "death": "#ff5d73", "ultimate": "#ffc857", "positioning": "#55e6a5"}
COMMON_METRICS: list[dict[str, Any]] = [
    {"key": "tracking_stability", "label": "트래킹 안정성", "category": "aim", "group": "common", "score_weight": 12},
    {"key": "aim_mobility_range", "label": "에임 가동 범위", "category": "aim", "group": "common", "score_weight": 8},
    {"key": "combat_cover_rate", "label": "교전 엄폐율", "category": "move", "group": "common", "score_weight": 10},
    {"key": "meaningless_jump_rate", "label": "무의미한 점프 빈도", "category": "move", "group": "common", "score_weight": 5},
    {"key": "scan_frequency", "label": "정보 스캔 빈도", "category": "judge", "group": "common", "score_weight": 8},
    {"key": "regroup_discipline", "label": "리그룹 이행률", "category": "op", "group": "common", "score_weight": 7},
    {"key": "fight_tempo_discipline", "label": "교전 템포 운영", "category": "op", "group": "common", "score_weight": 10},
]
ROLE_METRICS: dict[str, list[dict[str, Any]]] = {
    "dps": [
        {"key": "target_focus_priority", "label": "타겟 포커싱 우선순위", "category": "op", "group": "role", "score_weight": 12},
        {"key": "side_angle_occupancy", "label": "사이드 점유율", "category": "judge", "group": "role", "score_weight": 10},
        {"key": "effective_range_tempo", "label": "유효 사거리 템포", "category": "judge", "group": "role", "score_weight": 6},
        {"key": "ultimate_investment_efficiency", "label": "궁극기 투자 효율", "category": "op", "group": "role", "score_weight": 12},
    ],
    "tank": [
        {"key": "choke_control", "label": "길목 장악력", "category": "op", "group": "role", "score_weight": 14},
        {"key": "prefight_resource_preservation", "label": "교전 전 자원 보존율", "category": "op", "group": "role", "score_weight": 8},
        {"key": "skill_counted_entry", "label": "스킬 카운팅 진입", "category": "judge", "group": "role", "score_weight": 8},
        {"key": "aggro_pingpong_survival", "label": "어그로 핑퐁 생존", "category": "move", "group": "role", "score_weight": 10},
    ],
    "support": [
        {"key": "heal_damage_tempo_shift", "label": "공수 전환 템포", "category": "judge", "group": "role", "score_weight": 8},
        {"key": "critical_ally_reaction", "label": "치명상 반응 속도", "category": "move", "group": "role", "score_weight": 10},
        {"key": "self_survival_cooldown_preservation", "label": "생존기 보존 시간", "category": "op", "group": "role", "score_weight": 10},
        {"key": "survival_line_maintenance", "label": "생존 한계선 유지율", "category": "judge", "group": "role", "score_weight": 12},
    ],
}

st.set_page_config(page_title="Overwatch AI Coach MVP", layout="wide")


@dataclass
class VideoQuality:
    width: int
    height: int
    fps: float
    duration_sec: float
    grade: str
    notes: list[str]


@dataclass
class EventCandidate:
    candidate_id: str
    timestamp: str
    center_sec: float
    start_sec: float
    end_sec: float
    event_type: str
    motion_score: float


def normalize_enemy_comp_read(raw: dict[str, Any]) -> dict[str, Any]:
    dps = [str(value).strip() for value in raw.get("dps", []) if str(value).strip()]
    support = [str(value).strip() for value in raw.get("support", []) if str(value).strip()]
    return {
        "tank": str(raw.get("tank", "")).strip(),
        "dps": dps[:2],
        "support": support[:2],
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "evidence": str(raw.get("evidence", "")).strip(),
    }


def format_enemy_comp_read(enemy_comp: dict[str, Any]) -> str:
    tank = enemy_comp.get("tank") or "-"
    dps = ", ".join(enemy_comp.get("dps", [])) or "-"
    support = ", ".join(enemy_comp.get("support", [])) or "-"
    return f"Tank: {tank} / DPS: {dps} / Support: {support}"


def render_enemy_comp_context(context: str) -> str:
    return f"<p><b>적 조합 반영</b><br>{context}</p>" if context else ""
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Noto+Sans+KR:wght@400;500;700&display=swap');

        :root {
            --bg: #07111f;
            --bg-soft: #0c1c31;
            --panel: rgba(11, 24, 43, 0.92);
            --panel-alt: rgba(18, 38, 69, 0.9);
            --line: rgba(120, 178, 255, 0.22);
            --text: #e8f2ff;
            --muted: #8ea7c5;
            --cyan: #56c4ff;
            --teal: #65ffe2;
            --amber: #ffcf5d;
            --red: #ff6a78;
            --green: #65e7a4;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 15%, rgba(86,196,255,0.14), transparent 26rem),
                radial-gradient(circle at 85% 0%, rgba(101,255,226,0.10), transparent 24rem),
                linear-gradient(180deg, #07111f 0%, #09172b 48%, #06101d 100%);
            color: var(--text);
            font-family: 'Noto Sans KR', sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.75);
            backdrop-filter: blur(12px);
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, h4, p, label, span, div {
            color: var(--text);
        }

        .ow-header {
            background: linear-gradient(135deg, rgba(7,17,31,0.96), rgba(13,27,50,0.94));
            border: 1px solid var(--line);
            border-left: 4px solid var(--cyan);
            box-shadow: 0 24px 60px rgba(0,0,0,0.26);
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
        }

        .ow-kicker {
            font-family: 'Orbitron', sans-serif;
            color: var(--teal);
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
        }

        .ow-title {
            font-family: 'Orbitron', sans-serif;
            font-size: clamp(2rem, 4vw, 3.4rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-top: 0.35rem;
        }

        .ow-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin-top: 0.55rem;
        }

        .ow-card {
            background: linear-gradient(180deg, rgba(11,24,43,0.95), rgba(13,27,50,0.9));
            border: 1px solid var(--line);
            box-shadow: 0 14px 34px rgba(0,0,0,0.22);
            padding: 1rem 1.05rem;
            margin-bottom: 0.75rem;
        }

        .ow-card-title {
            font-family: 'Orbitron', sans-serif;
            color: var(--cyan);
            font-size: 0.92rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .ow-metric {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .ow-meta {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.35rem;
        }

        .lane-tag {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 0.45rem;
        }

        .lane-strength {
            background: rgba(101,231,164,0.12);
            color: var(--green);
            border: 1px solid rgba(101,231,164,0.3);
        }

        .lane-weakness {
            background: rgba(255,106,120,0.12);
            color: var(--red);
            border: 1px solid rgba(255,106,120,0.28);
        }

        .timeline-item {
            padding: 0.85rem 0.95rem;
            border: 1px solid var(--line);
            background: rgba(10, 20, 38, 0.72);
            margin-bottom: 0.6rem;
        }

        .timeline-title {
            font-weight: 700;
            font-size: 0.98rem;
        }

        .timeline-meta {
            color: var(--muted);
            font-size: 0.83rem;
            margin-top: 0.2rem;
        }

        .detail-panel {
            background: linear-gradient(180deg, rgba(9,18,33,0.94), rgba(13,27,50,0.92));
            border: 1px solid var(--line);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .detail-heading {
            font-family: 'Orbitron', sans-serif;
            color: var(--amber);
            font-size: 0.88rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .section-gap {
            margin-top: 1.15rem;
        }

        .stMultiSelect [data-baseweb="tag"] {
            background: rgba(86,196,255,0.14);
        }

        div[data-baseweb="select"] > div {
            background: rgba(11, 24, 43, 0.96) !important;
            border: 1px solid var(--line) !important;
            color: var(--text) !important;
        }

        div[data-baseweb="select"] div[role="combobox"],
        div[data-baseweb="select"] div[role="combobox"] *,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }

        div[data-baseweb="select"] svg {
            color: var(--text) !important;
            fill: var(--text) !important;
        }

        div[data-baseweb="popover"] ul {
            background: rgba(10, 20, 38, 0.98) !important;
            border: 1px solid var(--line) !important;
        }

        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] li *,
        div[data-baseweb="popover"] div[role="option"],
        div[data-baseweb="popover"] div[role="option"] * {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }

        div[data-baseweb="popover"] [aria-selected="true"] {
            background: rgba(86, 196, 255, 0.14) !important;
        }

        div[data-baseweb="popover"] [aria-selected="true"] *,
        div[data-baseweb="popover"] [data-highlighted="true"] * {
            color: var(--text) !important;
        }

        div[data-testid="stVideo"] video {
            border: 1px solid var(--line);
            background: #030812;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="ow-header">
            <div class="ow-kicker">AI REPLAY REVIEW</div>
            <div class="ow-title">Overwatch AI Coach MVP</div>
            <div class="ow-subtitle">잘한 장면과 보완 장면을 분리해 보고, 근거 프레임과 재생 구간까지 함께 복기하는 HUD형 코칭 대시보드</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_stat_card(title: str, value: str, meta: str = "") -> None:
    st.markdown(
        f"""
        <div class="ow-card">
            <div class="ow-card-title">{title}</div>
            <div class="ow-metric">{value}</div>
            <div class="ow-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return default


def get_oauth_client_config() -> dict[str, Any] | None:
    raw = get_secret("GOOGLE_OAUTH_CLIENT_JSON")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def get_oauth_token_secret() -> str:
    return get_secret("GOOGLE_OAUTH_TOKEN_JSON")


def get_oauth_token_scopes() -> list[str]:
    raw = get_oauth_token_secret().strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    scopes = payload.get("scopes", [])
    if isinstance(scopes, list):
        return [str(scope) for scope in scopes]
    if isinstance(scopes, str):
        return [scopes]
    return []


def token_secret_has_drive_scope() -> bool:
    return any(scope == DRIVE_SCOPES[0] for scope in get_oauth_token_scopes())


def token_secret_is_readonly_drive() -> bool:
    scopes = set(get_oauth_token_scopes())
    return "https://www.googleapis.com/auth/drive.readonly" in scopes and DRIVE_SCOPES[0] not in scopes


def token_secret_has_any_drive_access() -> bool:
    scopes = set(get_oauth_token_scopes())
    return bool(scopes & {"https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/drive.readonly"})


def is_cloud_runtime() -> bool:
    return bool(os.environ.get("STREAMLIT_SHARING_MODE")) or Path(__file__).as_posix().startswith("/mount/src/")


def extract_http_error_details(exc: HttpError) -> tuple[int | None, str, str]:
    status_code = getattr(getattr(exc, "resp", None), "status", None)
    reason = ""
    message = str(exc)
    content = getattr(exc, "content", b"")
    if not content:
        return status_code, reason, message
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status_code, reason, message
    error_payload = payload.get("error", {})
    if isinstance(error_payload, dict):
        message = str(error_payload.get("message") or message)
        errors = error_payload.get("errors", [])
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                reason = str(first.get("reason") or "")
    return status_code, reason, message


def render_drive_http_error(context_label: str, folder_id: str, exc: HttpError) -> None:
    status_code, reason, message = extract_http_error_details(exc)
    normalized_folder_id = normalize_drive_folder_id(folder_id)

    st.error(f"{context_label}에 실패했습니다. Google Drive 설정을 확인해 주세요.")
    st.caption(f"요청 폴더 ID: `{normalized_folder_id}`")
    if status_code:
        st.caption(f"응답 코드: `{status_code}`")
    if reason:
        st.caption(f"Drive reason: `{reason}`")
    st.caption(f"오류 메시지: `{message}`")

    hints: list[str] = []
    if status_code == 404 or reason == "notFound":
        hints.append("폴더 ID 또는 폴더 URL이 잘못되었거나, 현재 Google 계정으로 해당 폴더를 볼 수 없습니다.")
    if status_code == 401 or reason in {"authError", "invalidCredentials"}:
        hints.append("OAuth 토큰이 만료되었거나 잘못되었습니다. 로컬에서 새 token.json을 만든 뒤 Streamlit Secrets에 다시 넣어 주세요.")
    if status_code == 403 or reason in {"insufficientPermissions", "forbidden", "insufficientFilePermissions"}:
        hints.append("현재 Google 계정 또는 토큰 권한으로 이 폴더에 접근할 수 없습니다.")
    if not hints:
        hints.append("폴더 ID, OAuth 클라이언트 JSON, OAuth 토큰 JSON이 서로 같은 Google 계정 기준인지 확인해 주세요.")
    for hint in hints:
        st.write(f"- {hint}")
def render_oauth_refresh_error(context_label: str, exc: RefreshError) -> None:
    st.error(f"{context_label} 중 Google OAuth 토큰 갱신에 실패했습니다.")
    st.caption(f"오류 메시지: `{exc}`")
    st.write("- 배포 Secrets의 `GOOGLE_OAUTH_TOKEN_JSON`이 만료되었거나 폐기되었을 수 있습니다.")
    st.write("- Google OAuth 앱이 Testing 상태라면 refresh token이 자동 만료될 수 있습니다.")
    if token_secret_is_readonly_drive():
        st.write("- 현재 토큰은 읽기 전용입니다. 자동 저장은 하지 않고, 결과 JSON을 직접 다운로드해 Drive에 올려 주세요.")
    else:
        st.write("- 새 토큰 JSON에는 유효한 `refresh_token`이 포함되어야 합니다.")
def load_oauth_credentials() -> Credentials | None:
    token_secret = get_oauth_token_secret()
    effective_scopes = get_oauth_token_scopes() or DRIVE_SCOPES
    if token_secret:
        try:
            creds = Credentials.from_authorized_user_info(json.loads(token_secret), effective_scopes)
        except Exception:
            creds = None
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                raise
            except Exception:
                return None
        if creds and creds.valid:
            return creds
    if not TOKEN_PATH.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), effective_scopes)
    except Exception:
        return None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_oauth_credentials(creds)
        except RefreshError:
            raise
        except Exception:
            return None
    return creds if creds and creds.valid else None


def save_oauth_credentials(creds: Credentials) -> None:
    if get_oauth_token_secret():
        return
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")


def clear_oauth_credentials() -> None:
    if get_oauth_token_secret():
        return
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()


def authorize_drive_oauth() -> None:
    config = get_oauth_client_config()
    if not config:
        st.error("`GOOGLE_OAUTH_CLIENT_JSON` 시크릿이 필요합니다.")
        st.stop()
    flow = InstalledAppFlow.from_client_config(config, DRIVE_SCOPES)
    try:
        creds = flow.run_local_server(host="localhost", port=0, open_browser=True)
    except webbrowser.Error:
        st.error("배포된 Streamlit 앱에서는 로컬 브라우저 OAuth 로그인을 직접 열 수 없습니다.")
        st.write("- 이 버튼은 로컬 PC에서만 사용 가능합니다.")
        st.write("- 로컬에서 로그인해 새 `token.json`을 만든 뒤, 그 내용을 배포 Secrets의 `GOOGLE_OAUTH_TOKEN_JSON`에 넣어주세요.")
        st.stop()
    save_oauth_credentials(creds)


def get_drive_service():
    config = get_oauth_client_config()
    if not config:
        st.error("`GOOGLE_OAUTH_CLIENT_JSON` 시크릿이 없습니다. Streamlit Secrets에 OAuth 클라이언트 JSON을 넣어 주세요.")
        st.stop()

    if get_oauth_token_secret().strip() and not token_secret_has_any_drive_access():
        st.error("배포된 OAuth 토큰에 Google Drive 접근 권한이 없습니다.")
        st.write("- 로컬에서 Google 로그인 후 새 토큰을 발급해 주세요.")
        st.write("- 영상 불러오기만 하려면 `drive.readonly` 권한으로도 충분합니다.")
        st.stop()

    try:
        creds = load_oauth_credentials()
    except RefreshError as exc:
        render_oauth_refresh_error("Google Drive 연결 준비", exc)
        st.stop()

    if not creds:
        st.error("배포된 앱에 유효한 Drive OAuth 토큰이 없습니다.")
        st.write("- Streamlit Cloud에서는 브라우저 로그인 버튼을 사용할 수 없습니다.")
        st.write("- 로컬 PC에서 새 `token.json`을 만든 뒤, 그 JSON 전체를 `GOOGLE_OAUTH_TOKEN_JSON`에 붙여넣어 주세요.")
        st.stop()

    return build("drive", "v3", credentials=creds)
def normalize_drive_folder_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if "/folders/" in raw:
        parsed = urllib.parse.urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        try:
            folder_index = parts.index("folders")
            return parts[folder_index + 1]
        except (ValueError, IndexError):
            return raw
    return raw


def drive_folder_url(value: str) -> str:
    raw = (value or "").strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    folder_id = normalize_drive_folder_id(raw)
    return f"https://drive.google.com/drive/folders/{folder_id}" if folder_id else ""


def render_drive_shortcuts(input_folder_id: str, output_folder_id: str) -> None:
    left, right = st.columns(2)
    input_url = drive_folder_url(input_folder_id)
    output_url = drive_folder_url(output_folder_id)
    with left:
        if input_url:
            st.link_button("원본 영상 업로드 폴더 열기", input_url, use_container_width=True)
    with right:
        if output_url:
            st.link_button("분석 결과 저장 폴더 열기", output_url, use_container_width=True)


def list_videos(drive_service, folder_id: str) -> list[dict[str, Any]]:
    folder_id = normalize_drive_folder_id(folder_id)
    query = f"'{folder_id}' in parents and trashed=false and mimeType contains 'video/'"
    response = (
        drive_service.files()
        .list(q=query, fields="files(id,name,mimeType,size,createdTime)", orderBy="createdTime desc", pageSize=100)
        .execute()
    )
    return response.get("files", [])


def list_saved_reports(drive_service, folder_id: str) -> list[dict[str, Any]]:
    folder_id = normalize_drive_folder_id(folder_id)
    query = f"'{folder_id}' in parents and trashed=false and mimeType='application/json'"
    response = (
        drive_service.files()
        .list(q=query, fields="files(id,name,mimeType,size,createdTime)", orderBy="createdTime desc", pageSize=100)
        .execute()
    )
    return response.get("files", [])


def download_json_file(drive_service, file_id: str) -> dict[str, Any]:
    request = drive_service.files().get_media(fileId=file_id)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = Path(tmp.name)
    tmp.close()
    with tmp_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return json.loads(tmp_path.read_text(encoding="utf-8"))


def count_reports_this_kst_week(saved_reports: list[dict[str, Any]]) -> int:
    kst = ZoneInfo("Asia/Seoul")
    now = datetime.now(kst)
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)
    total = 0
    for item in saved_reports:
        created_at = item.get("createdTime")
        if not created_at:
            continue
        try:
            created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).astimezone(kst)
        except ValueError:
            continue
        if week_start <= created_dt < week_end:
            total += 1
    return total


def download_video(drive_service, file_id: str, file_name: str) -> Path:
    suffix = Path(file_name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    tmp.close()

    request = drive_service.files().get_media(fileId=file_id)
    with tmp_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return tmp_path


def compute_file_hash(video_path: Path) -> str:
    digest = hashlib.sha256()
    with video_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(video_path: Path) -> VideoQuality:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoQuality(0, 0, 0.0, 0.0, "C", ["鍮꾨뵒?ㅻ? ?????놁뒿?덈떎."])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    notes: list[str] = []
    score = 0
    if width >= 1920 and height >= 1080:
        score += 2
    else:
        notes.append("沅뚯옣 ?댁긽??1080p)蹂대떎 ??뒿?덈떎.")
    if fps >= 50:
        score += 2
    elif fps >= 30:
        score += 1
        notes.append("60fps 권장 대비 낮아 에임 분석 정밀도가 다소 떨어질 수 있습니다.")
    else:
        notes.append("fps가 낮아 분석 신뢰도가 크게 떨어질 수 있습니다.")
    if duration >= 180:
        score += 1
    else:
        notes.append("영상 길이가 짧아 상황 다양성이 부족할 수 있습니다.")

    grade = "A" if score >= 4 else "B" if score >= 2 else "C"
    return VideoQuality(width, height, fps, duration, grade, notes)


def format_seconds_to_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_timestamp_to_seconds(value: str) -> float:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", value or "")
    if not match:
        return 0.0
    return float(int(match.group(1)) * 60 + int(match.group(2)))


def sample_frames(video_path: Path, sample_seconds: list[float]) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 else 0.0
    frames: list[dict[str, Any]] = []

    for raw_sec in sample_seconds:
        sec = max(0.0, min(duration, float(raw_sec)))
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok2, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            continue
        frames.append({"timestamp": sec, "rgb": rgb, "jpg_bytes": encoded.tobytes()})

    cap.release()
    return frames


def detect_event_candidates(video_path: Path, window_step_sec: float = 1.5, max_candidates: int = 8) -> list[EventCandidate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 else 0.0
    timestamps = np.arange(0.0, max(duration, window_step_sec), window_step_sec)
    prev_gray = None
    scored: list[dict[str, float]] = []

    for sec in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        resized = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            edge = cv2.Canny(gray, 80, 160)
            motion_score = float(np.mean(diff) * 0.75 + np.mean(edge) * 0.25)
            scored.append({"center_sec": float(sec), "motion_score": motion_score})
        prev_gray = gray

    cap.release()
    scored.sort(key=lambda item: item["motion_score"], reverse=True)
    selected: list[dict[str, float]] = []
    for item in scored:
        if any(abs(item["center_sec"] - kept["center_sec"]) < 7.0 for kept in selected):
            continue
        selected.append(item)
        if len(selected) >= max_candidates:
            break
    selected.sort(key=lambda item: item["center_sec"])

    candidates: list[EventCandidate] = []
    for index, item in enumerate(selected, start=1):
        center_sec = item["center_sec"]
        start_sec = max(0.0, center_sec - 5.0)
        end_sec = min(duration, center_sec + 5.0)
        event_type = "engagement"
        if item["motion_score"] > 45:
            event_type = "ultimate"
        if index == len(selected) and duration - center_sec < 12:
            event_type = "death"
        candidates.append(
            EventCandidate(
                candidate_id=f"E{index}",
                timestamp=format_seconds_to_mmss(center_sec),
                center_sec=center_sec,
                start_sec=start_sec,
                end_sec=end_sec,
                event_type=event_type,
                motion_score=float(item["motion_score"]),
            )
        )
    return candidates


def collect_candidate_frames(video_path: Path, candidates: list[EventCandidate]) -> dict[str, list[dict[str, Any]]]:
    frame_map: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        frame_map[candidate.candidate_id] = sample_frames(
            video_path,
            [candidate.center_sec - 2.0, candidate.center_sec, candidate.center_sec + 2.0],
        )
    return frame_map


def collect_candidate_clips(
    video_path: Path,
    candidates: list[EventCandidate],
    target_width: int = 960,
    target_fps: float = 12.0,
) -> dict[str, bytes]:
    clip_map: dict[str, bytes] = {}
    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        return clip_map

    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(probe.get(cv2.CAP_PROP_FPS) or 30.0)
    probe.release()

    if source_width <= 0 or source_height <= 0:
        return clip_map

    scale = min(1.0, target_width / float(source_width)) if target_width > 0 else 1.0
    clip_width = max(2, int(round(source_width * scale)))
    clip_height = max(2, int(round(source_height * scale)))
    if clip_width % 2:
        clip_width += 1
    if clip_height % 2:
        clip_height += 1

    output_fps = max(6.0, min(target_fps, source_fps if source_fps > 0 else target_fps))
    frame_skip = max(1, int(round((source_fps or output_fps) / output_fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for candidate in candidates:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, candidate.start_sec) * 1000.0)

        tmp_path: Path | None = None
        writer: cv2.VideoWriter | None = None
        frame_index = 0
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp_path = Path(tmp.name)
            writer = cv2.VideoWriter(str(tmp_path), fourcc, output_fps, (clip_width, clip_height))
            if not writer.isOpened():
                continue

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                current_sec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if current_sec > candidate.end_sec + 0.05:
                    break
                if frame_index % frame_skip == 0:
                    if scale != 1.0:
                        frame = cv2.resize(frame, (clip_width, clip_height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if tmp_path is None or not tmp_path.exists():
            continue
        try:
            clip_map[candidate.candidate_id] = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    return clip_map


def get_metric_catalog(role: str) -> list[dict[str, Any]]:
    return COMMON_METRICS + ROLE_METRICS[role]


def build_metric_defaults(role: str, score_weights: dict[str, int]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for metric in get_metric_catalog(role):
        category_key = metric["category"]
        defaults[metric["key"]] = {
            "label": metric["label"],
            "category": category_key,
            "group": metric["group"],
            "score": 0,
            "score_weight": metric["score_weight"],
            "category_weight": round(score_weights[category_key] / 100.0, 2),
            "evaluation": "",
            "feedback_direction": "",
            "confidence": 0.0,
            "low_confidence_reason": "",
        }
    return defaults


def build_candidates_from_result(result: dict[str, Any]) -> list[EventCandidate]:
    raw_candidates = result.get("input_video", {}).get("event_candidates", [])
    candidates: list[EventCandidate] = []
    for item in raw_candidates:
        try:
            candidates.append(
                EventCandidate(
                    candidate_id=str(item["candidate_id"]),
                    timestamp=str(item["timestamp"]),
                    center_sec=float(item["center_sec"]),
                    start_sec=float(item["start_sec"]),
                    end_sec=float(item["end_sec"]),
                    event_type=str(item["event_type"]),
                    motion_score=float(item.get("motion_score", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return candidates


def build_radar_chart(scores: dict[str, int]) -> go.Figure:
    categories = [CATEGORY_LABELS[key] for key in ["aim", "move", "judge", "op"]]
    values = [scores.get(key, 0) for key in ["aim", "move", "judge", "op"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="#65ffe2", width=3),
            fillcolor="rgba(101,255,226,0.22)",
            name="score",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(color="#d9ecff")), angularaxis=dict(tickfont=dict(color="#d9ecff"))),
        showlegend=False,
        height=300,
    )
    return fig


def normalize_result(
    raw: dict[str, Any],
    role: str,
    hero: str,
    quality: VideoQuality,
    video_hash: str,
    candidates: list[EventCandidate],
) -> dict[str, Any]:
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    weights = ROLE_WEIGHTS[role]
    scores = {
        "aim": int(raw.get("scores", {}).get("aim", 0)),
        "move": int(raw.get("scores", {}).get("move", 0)),
        "judge": int(raw.get("scores", {}).get("judge", 0)),
        "op": int(raw.get("scores", {}).get("op", 0)),
    }
    weighted_score = int(round(sum(scores[key] * weights[key] / 100 for key in scores)))
    confidence_score = float(raw.get("confidence_score", 0.0) or 0.0)

    def normalize_items(items: list[dict[str, Any]], lane: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in items:
            candidate_id = str(item.get("candidate_id", "")).strip()
            if candidate_id not in candidate_map:
                continue
            candidate = candidate_map[candidate_id]
            normalized.append(
                {
                    "lane": lane,
                    "candidate_id": candidate_id,
                    "timestamp": candidate.timestamp,
                    "event_type": item.get("event_type", candidate.event_type),
                    "category": item.get("category", "judge"),
                    "metric_name": item.get("metric_name", CATEGORY_LABELS.get(item.get("category", "judge"), "Judge")),
                    "summary": item.get("summary", ""),
                    "evaluation_basis": item.get("evaluation_basis", ""),
                    "feedback_direction": item.get("feedback_direction", ""),
                    "primary_cause": item.get("primary_cause", ""),
                    "secondary_cause": item.get("secondary_cause", ""),
                    "enemy_comp_context": item.get("enemy_comp_context", ""),
                    "action_item": item.get("action_item", ""),
                    "priority": item.get("priority", "mid"),
                    "confidence": float(item.get("confidence", confidence_score) or confidence_score),
                }
            )
        return normalized

    strengths = normalize_items(raw.get("strength_items", []), "strength")
    weaknesses = normalize_items(raw.get("weakness_items", []), "weakness")
    death_causes = []
    for item in raw.get("death_causes", []):
        candidate_id = str(item.get("candidate_id", "")).strip()
        candidate = candidate_map.get(candidate_id)
        if not candidate:
            continue
        death_causes.append(
            {
                "candidate_id": candidate_id,
                "timestamp": candidate.timestamp,
                "primary": item.get("primary", ""),
                "secondary": item.get("secondary", ""),
                "enemy_comp_context": item.get("enemy_comp_context", ""),
                "confidence": float(item.get("confidence", confidence_score) or confidence_score),
            }
        )

    metric_defaults = build_metric_defaults(role, weights)
    raw_metrics = raw.get("metrics", {})
    for metric_key, default_value in metric_defaults.items():
        incoming = raw_metrics.get(metric_key, {})
        metric_defaults[metric_key] = {
            **default_value,
            "score": int(incoming.get("score", default_value["score"])),
            "evaluation": incoming.get("evaluation", default_value["evaluation"]),
            "feedback_direction": incoming.get("feedback_direction", default_value["feedback_direction"]),
            "confidence": float(incoming.get("confidence", default_value["confidence"]) or default_value["confidence"]),
            "low_confidence_reason": incoming.get("low_confidence_reason", default_value["low_confidence_reason"]),
        }

    score_groups = {
        "common": sum(metric_defaults[key]["score"] * metric_defaults[key]["score_weight"] / 60 for key in metric_defaults if metric_defaults[key]["group"] == "common"),
        "role": sum(metric_defaults[key]["score"] * metric_defaults[key]["score_weight"] / 40 for key in metric_defaults if metric_defaults[key]["group"] == "role"),
    }

    return {
        "schema_version": raw.get("schema_version", "1.4.0"),
        "model_version": raw.get("model_version", MODEL_NAME),
        "source_video_hash": video_hash,
        "confidence_score": confidence_score,
        "meta": {
            "hero": hero,
            "role": role.upper(),
            "confidence_score": confidence_score,
            "quality_grade": quality.grade,
            "duration_sec": round(quality.duration_sec, 2),
        },
        "weights": weights,
        "scores": scores,
        "weighted_score": weighted_score,
        "metrics": metric_defaults,
        "score_groups": {
            "common_60": int(round(score_groups["common"] * 0.6)),
            "role_40": int(round(score_groups["role"] * 0.4)),
        },
        "strength_items": strengths,
        "weakness_items": weaknesses,
        "death_causes": death_causes,
        "recommended_focus": raw.get("recommended_focus", []),
        "recommended_focus_guides": raw.get("recommended_focus_guides", []),
        "enemy_comp_read": normalize_enemy_comp_read(raw.get("enemy_comp_read", {})),
        "notes": raw.get("notes", []),
    }


def build_analysis_prompt(
    hero: str,
    role: str,
    weights: dict[str, int],
    quality: VideoQuality,
    candidate_summary: list[dict[str, Any]],
    video_hash: str,
    metric_catalog: list[dict[str, Any]],
    metric_schema: dict[str, Any],
) -> str:
    return f"""
You are an expert Overwatch coach analyzing a POV replay.
The player's hero is {hero} and the role is {role}.

Use only the provided event candidates and frames. Do not invent scenes, timestamps, hero picks, ult usage, or cooldown facts that are not visually supported.
Return JSON only, and write all user-facing text in Korean.

Your job:
- Separate 2-3 strength scenes and 2-3 weakness scenes.
- Keep every scene anchored to one provided candidate_id.
- Infer the likely enemy team composition from visible evidence when possible.
- Use that inferred enemy composition to make feedback more specific, especially for death causes, target priority, spacing, cover timing, angle discipline, and cooldown usage.
- If enemy composition is unclear, leave uncertain slots blank and lower confidence instead of guessing.
- Make the improvement guidance kind, practical, and easy to apply in the very next fight.

Input metadata:
- model_version: {MODEL_NAME}
- role_weights: {json.dumps(weights, ensure_ascii=False)}
- quality_grade: {quality.grade}
- candidate_list: {json.dumps(candidate_summary, ensure_ascii=False)}
- source_video_hash: {video_hash}
- metric_catalog: {json.dumps(metric_schema, ensure_ascii=False)}

Return this JSON shape:
{{
  "schema_version": "1.4.0",
  "model_version": "{MODEL_NAME}",
  "confidence_score": 0.0,
  "scores": {{"aim": 0, "move": 0, "judge": 0, "op": 0}},
  "metrics": {json.dumps({metric["key"]: {"score": 0, "evaluation": "...", "feedback_direction": "...", "confidence": 0.0, "low_confidence_reason": "..."} for metric in metric_catalog}, ensure_ascii=False)},
  "enemy_comp_read": {{
    "tank": "...",
    "dps": ["...", "..."],
    "support": ["...", "..."],
    "confidence": 0.0,
    "evidence": "..."
  }},
  "strength_items": [
    {{
      "candidate_id": "E1",
      "event_type": "engagement|ultimate|positioning",
      "category": "aim|move|judge|op",
      "metric_name": "...",
      "summary": "...",
      "evaluation_basis": "...",
      "feedback_direction": "...",
      "primary_cause": "...",
      "secondary_cause": "...",
      "enemy_comp_context": "...",
      "action_item": "...",
      "priority": "high|mid|low",
      "confidence": 0.0
    }}
  ],
  "weakness_items": [
    {{
      "candidate_id": "E2",
      "event_type": "engagement|death|ultimate|positioning",
      "category": "aim|move|judge|op",
      "metric_name": "...",
      "summary": "...",
      "evaluation_basis": "...",
      "feedback_direction": "...",
      "primary_cause": "...",
      "secondary_cause": "...",
      "enemy_comp_context": "...",
      "action_item": "...",
      "priority": "high|mid|low",
      "confidence": 0.0
    }}
  ],
  "death_causes": [
    {{
      "candidate_id": "E2",
      "primary": "...",
      "secondary": "...",
      "enemy_comp_context": "...",
      "confidence": 0.0
    }}
  ],
  "recommended_focus": ["...", "..."],
  "recommended_focus_guides": [
    {{
      "title": "...",
      "why_it_matters": "...",
      "how_to_apply": "...",
      "enemy_comp_context": "..."
    }}
  ],
  "notes": ["..."]
}}

Rules:
- strength_items must be 2-3 items and weakness_items must be 2-3 items.
- death_causes must be 1-2 items and only reference weakness candidates.
- recommended_focus should be short headline bullets.
- recommended_focus_guides should expand those headlines into warmer and more concrete coaching guidance.
- enemy_comp_context should explain how the inferred enemy composition changes the correct decision in that scene.
- If the enemy composition read is weak, mention the uncertainty briefly in enemy_comp_read.evidence or low_confidence_reason.
- confidence must stay within 0.0-1.0.
""".strip()


def gemini_analyze(
    api_key: str,
    role: str,
    hero: str,
    quality: VideoQuality,
    candidates: list[EventCandidate],
    frame_map: dict[str, list[dict[str, Any]]],
    video_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    client = Client(api_key=api_key)
    weights = ROLE_WEIGHTS[role]
    metric_catalog = get_metric_catalog(role)
    candidate_summary = [
        {
            "candidate_id": candidate.candidate_id,
            "timestamp": candidate.timestamp,
            "event_type": candidate.event_type,
            "window": f"{candidate.start_sec:.1f}-{candidate.end_sec:.1f}",
            "motion_score": round(candidate.motion_score, 2),
        }
        for candidate in candidates
    ]
    metric_schema = {
        metric["key"]: {
            "label": metric["label"],
            "category": metric["category"],
            "group": metric["group"],
            "score_weight": metric["score_weight"],
            "required_fields": ["score", "evaluation", "feedback_direction", "confidence", "low_confidence_reason"],
        }
        for metric in metric_catalog
    }
    prompt = build_analysis_prompt(
        hero=hero,
        role=role,
        weights=weights,
        quality=quality,
        candidate_summary=candidate_summary,
        video_hash=video_hash,
        metric_catalog=metric_catalog,
        metric_schema=metric_schema,
    )

    contents: list[Any] = [prompt]
    for candidate in candidates:
        contents.append(f"{candidate.candidate_id} timestamp={candidate.timestamp} event_type={candidate.event_type}")
        for frame in frame_map[candidate.candidate_id]:
            contents.append(f"{candidate.candidate_id}_frame timestamp={frame['timestamp']:.1f}s")
            contents.append(types.Part.from_bytes(data=frame["jpg_bytes"], mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.15),
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()
    raw = json.loads(raw_text)

    usage = {}
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        usage = {
            "prompt_token_count": getattr(usage_metadata, "prompt_token_count", 0) or 0,
            "candidates_token_count": getattr(usage_metadata, "candidates_token_count", 0) or 0,
            "thoughts_token_count": getattr(usage_metadata, "thoughts_token_count", 0) or 0,
            "total_token_count": getattr(usage_metadata, "total_token_count", 0) or 0,
        }
    return normalize_result(raw, role, hero, quality, video_hash, candidates), usage

def upload_json_result(drive_service, output_folder_id: str, filename: str, payload: dict[str, Any]) -> str:
    output_folder_id = normalize_drive_folder_id(output_folder_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    metadata = {"name": filename, "parents": [output_folder_id], "mimeType": "application/json"}
    media = MediaFileUpload(str(tmp_path), mimetype="application/json")
    created = drive_service.files().create(body=metadata, media_body=media, fields="id,webViewLink").execute()
    return created.get("webViewLink", "")


def estimate_cost_krw(usage: dict[str, Any]) -> float:
    usd_to_krw = 1509.0
    input_usd = (float(usage.get("prompt_token_count", 0)) / 1_000_000) * 1.50
    output_usd = ((float(usage.get("candidates_token_count", 0)) + float(usage.get("thoughts_token_count", 0))) / 1_000_000) * 9.00
    return round((input_usd + output_usd) * usd_to_krw, 2)


def render_timeline_item(item: dict[str, Any]) -> None:
    lane_class = "lane-strength" if item["lane"] == "strength" else "lane-weakness"
    lane_text = LANE_LABELS[item["lane"]]
    event_color = TIMELINE_COLORS.get(item["event_type"], "#4db6ff")
    st.markdown(
        f"""
        <div class="timeline-item" style="border-left: 5px solid {event_color};">
            <div class="timeline-title"><span class="lane-tag {lane_class}">{lane_text}</span>{item['timestamp']} | {item['metric_name']}</div>
            <div style="margin-top: 0.25rem;">{item['summary']}</div>
            <div class="timeline-meta">{item['candidate_id']} | {item['event_type']} | priority={item['priority']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_panel(
    title: str,
    item: dict[str, Any],
    candidate_map: dict[str, EventCandidate],
    frame_map: dict[str, list[dict[str, Any]]],
    clip_map: dict[str, bytes],
) -> None:
    candidate = candidate_map[item["candidate_id"]]
    frames = frame_map.get(item["candidate_id"], [])
    clip_bytes = clip_map.get(item["candidate_id"])
    left, right = st.columns([1.05, 1.25])
    with left:
        st.markdown(f"<div class='detail-panel'><div class='detail-heading'>{title}</div>", unsafe_allow_html=True)
        if clip_bytes:
            st.video(clip_bytes, format="video/mp4", muted=True)
        else:
            st.info(f"{candidate.timestamp} 전후 10초 클립을 준비하지 못해 프레임 근거만 표시합니다.")
        if frames:
            frame_cols = st.columns(len(frames))
            for index, frame in enumerate(frames):
                with frame_cols[index]:
                    st.image(frame["rgb"], caption=f"{frame['timestamp']:.1f}초", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            f"""
            <div class='detail-panel'>
                <div class='detail-heading'>{item['timestamp']} | {item['metric_name']}</div>
                <p><b>장면 요약</b><br>{item['summary']}</p>
                <p><b>평가 근거</b><br>{item['evaluation_basis']}</p>
                <p><b>피드백 방향</b><br>{item['feedback_direction']}</p>
                <p><b>주원인</b><br>{item['primary_cause']}</p>
                <p><b>보조원인</b><br>{item['secondary_cause']}</p>
                <p><b>실행 루틴</b><br>{item['action_item']}</p>
                <p><b>신뢰도</b><br>{item['confidence']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
def render_report(
    result: dict[str, Any],
    quality: VideoQuality,
    candidates: list[EventCandidate],
    frame_map: dict[str, list[dict[str, Any]]],
    clip_map: dict[str, bytes],
) -> None:
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    scores = result["scores"]
    confidence_score = float(result.get("confidence_score", 0.0))
    usage = result.get("input_video", {}).get("usage", {})
    estimate_krw = float(result.get("input_video", {}).get("estimated_cost_krw", 0.0))
    timeline_items = result.get("strength_items", []) + result.get("weakness_items", [])
    timeline_items.sort(key=lambda item: parse_timestamp_to_seconds(item["timestamp"]))
    score_groups = result.get("score_groups", {})

    score_col, reli_col, cost_col, time_col = st.columns(4)
    with score_col:
        render_stat_card("모델", MODEL_LABEL, result.get("model_version", MODEL_NAME))
    with reli_col:
        render_stat_card("신뢰도", f"{confidence_score * 100:.0f}%", f"입력 품질 {quality.grade}")
    with cost_col:
        render_stat_card("예상 비용", f"{estimate_krw:,.0f}원", f"입력 {usage.get('prompt_token_count', 0)} / 출력 {usage.get('candidates_token_count', 0)}")
    with time_col:
        render_stat_card("점수 구성", f"{score_groups.get('common_60', 0)} + {score_groups.get('role_40', 0)}", "공통 60점 + 포지션 특화 40점")

    chart_col, meta_col = st.columns([1.0, 1.1])
    with chart_col:
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.plotly_chart(build_radar_chart(scores), use_container_width=True)
    with meta_col:
        st.markdown("<div class='ow-card'><div class='ow-card-title'>세션 메타</div>", unsafe_allow_html=True)
        st.markdown(
            f"**영상 길이**  \n{quality.duration_sec / 60:.1f}분  \n**해상도**  \n{quality.width}x{quality.height} / {quality.fps:.1f}fps"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    enemy_comp_read = result.get("enemy_comp_read", {})
    if any([enemy_comp_read.get("tank"), enemy_comp_read.get("dps"), enemy_comp_read.get("support")]):
        st.markdown(
            f"""
            <div class='ow-card'>
                <div class='ow-card-title'>적 조합 추정</div>
                <div>{format_enemy_comp_read(enemy_comp_read)}</div>
                <div class='ow-meta'>신뢰도={enemy_comp_read.get('confidence', 0.0)} | 근거: {enemy_comp_read.get('evidence', '-') or '-'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("공통 지표 60점")
    for metric in [result["metrics"][item["key"]] for item in COMMON_METRICS]:
        st.markdown(
            f"""
            <div class="ow-card">
                <div class="ow-card-title">{metric['label']}</div>
                <div><b>점수</b> {metric.get('score', 0)} / 가중치 {metric.get('score_weight', 0)}</div>
                <div><b>평가 방식</b><br>{metric.get('evaluation', '-')}</div>
                <div style="margin-top:0.35rem;"><b>피드백 방향</b><br>{metric.get('feedback_direction', '-')}</div>
                <div class="ow-meta">신뢰도={metric.get('confidence', 0.0)} | 낮은 신뢰도 이유: {metric.get('low_confidence_reason', '') or '-'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("포지션 특화 40점")
    role_metric_defs = ROLE_METRICS[result["meta"]["role"].lower()]
    for item in role_metric_defs:
        metric = result["metrics"][item["key"]]
        st.markdown(
            f"""
            <div class="ow-card">
                <div class="ow-card-title">{metric['label']}</div>
                <div><b>점수</b> {metric.get('score', 0)} / 가중치 {metric.get('score_weight', 0)}</div>
                <div><b>평가 방식</b><br>{metric.get('evaluation', '-')}</div>
                <div style="margin-top:0.35rem;"><b>피드백 방향</b><br>{metric.get('feedback_direction', '-')}</div>
                <div class="ow-meta">신뢰도={metric.get('confidence', 0.0)} | 낮은 신뢰도 이유: {metric.get('low_confidence_reason', '') or '-'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("이벤트 타임라인")
    filter_left, filter_mid, filter_right = st.columns(3)
    lane_filter = filter_left.multiselect("피드백 구간", options=["strength", "weakness"], default=["strength", "weakness"], format_func=lambda value: LANE_LABELS[value], key="lane_filter")
    category_filter = filter_mid.multiselect("평가 영역", options=["aim", "move", "judge", "op"], default=["aim", "move", "judge", "op"], format_func=lambda value: CATEGORY_LABELS[value], key="category_filter")
    event_filter = filter_right.multiselect("이벤트 유형", options=["engagement", "death", "ultimate", "positioning"], default=["engagement", "death", "ultimate", "positioning"], key="event_filter")

    filtered_items = [
        item for item in timeline_items if item["lane"] in lane_filter and item["category"] in category_filter and item["event_type"] in event_filter
    ]
    if not filtered_items:
        st.info("현재 필터에 맞는 이벤트가 없습니다.")
    for item in filtered_items:
        render_timeline_item(item)

    st.subheader("잘한 장면")
    for index, item in enumerate([it for it in filtered_items if it["lane"] == "strength"], start=1):
        render_detail_panel(f"잘한 장면 {index}", item, candidate_map, frame_map, clip_map)

    st.subheader("보완 장면")
    for index, item in enumerate([it for it in filtered_items if it["lane"] == "weakness"], start=1):
        render_detail_panel(f"보완 장면 {index}", item, candidate_map, frame_map, clip_map)

    st.subheader("데스 원인")
    for cause in result.get("death_causes", []):
        st.markdown(
            f"""
            <div class="ow-card">
                <div class="ow-card-title">{cause['timestamp']} | 데스 원인</div>
                <div><b>주원인</b><br>{cause['primary']}</div>
                <div style="margin-top:0.45rem;"><b>보조원인</b><br>{cause['secondary']}</div>
                <div class="ow-meta">candidate={cause['candidate_id']} / confidence={cause['confidence']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result.get("recommended_focus"):
        st.subheader("우선 개선 과제")
        for focus in result["recommended_focus"]:
            st.markdown(f"- {focus}")

    if result.get("recommended_focus_guides"):
        st.subheader("개선 실행 가이드")
        for guide in result["recommended_focus_guides"]:
            st.markdown(
                f"""
                <div class="ow-card">
                    <div class="ow-card-title">{guide.get('title', '-')}</div>
                    <div><b>왜 중요한가</b><br>{guide.get('why_it_matters', '-')}</div>
                    <div style="margin-top:0.45rem;"><b>다음 교전 적용법</b><br>{guide.get('how_to_apply', '-')}</div>
                    {render_enemy_comp_context(guide.get('enemy_comp_context', ''))}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info("분석 결과를 구글드라이브에 저장하면, 이후 다시 확인할 수 있습니다.")
    st.info("자동 저장은 사용하지 않습니다. 아래 JSON 다운로드 버튼으로 결과를 저장한 뒤, 직접 Google Drive 결과 폴더에 올려 주세요.")

    st.download_button(
        "결과 JSON 다운로드",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"analysis_{result.get('input_video', {}).get('name', 'report')}.json",
        mime="application/json",
    )


def main() -> None:
    inject_css()
    render_header()

    input_folder_id = get_secret("DRIVE_INPUT_FOLDER_ID")
    output_folder_id = get_secret("DRIVE_OUTPUT_FOLDER_ID")
    gemini_api_key = get_secret("GEMINI_API_KEY")
    if not input_folder_id or not output_folder_id or not gemini_api_key:
        st.warning("필수 시크릿 값이 비어 있습니다. README의 secrets 예시를 채워 주세요.")
        st.stop()

    drive = get_drive_service()
    try:
        videos = list_videos(drive, input_folder_id)
    except HttpError as exc:
        render_drive_http_error("입력 폴더 영상 목록 조회", input_folder_id, exc)
        st.stop()
    if not videos:
        st.info("입력 폴더에 분석 가능한 영상 파일이 없습니다.")
        st.stop()
    try:
        saved_reports = list_saved_reports(drive, output_folder_id)
    except HttpError as exc:
        render_drive_http_error("결과 폴더 JSON 목록 조회", output_folder_id, exc)
        saved_reports = []
    weekly_report_count = count_reports_this_kst_week(saved_reports)

    usage_col, saved_col = st.columns(2)
    with usage_col:
        render_stat_card("금주 분석 횟수", str(weekly_report_count), "한국시간 기준 월요일 00:00부터 집계합니다.")
    with saved_col:
        render_stat_card("저장된 결과 수", str(len(saved_reports)), "결과 폴더에 있는 JSON 파일 기준입니다.")

    video_by_id = {video["id"]: video for video in videos}
    video_by_name = {video["name"]: video for video in videos}
    mode = st.radio("작업 모드", options=["새로 분석", "저장된 결과 다시보기"], horizontal=True)

    render_drive_shortcuts(input_folder_id, output_folder_id)
    st.caption("원본 녹화 영상은 입력 폴더에, 다운로드한 분석 결과 JSON은 결과 폴더에 직접 올려 주세요.")
    st.info("분석 결과를 구글드라이브에 저장하면, 이후 다시 확인할 수 있습니다.")

    if mode == "새로 분석":
        top_left, top_right, top_third = st.columns([1.6, 1.0, 1.0])
        with top_left:
            selected = st.selectbox("분석할 영상 선택", options=videos, format_func=lambda item: item["name"])
        with top_right:
            role = st.selectbox("역할", ["tank", "dps", "support"])
        with top_third:
            hero = st.selectbox("영웅", ROLE_HEROES[role])

        if not st.button("분석 시작", type="primary"):
            return

        with st.status("분석 진행 중", expanded=True) as status:
            st.write("1) Google Drive에서 원본 영상을 불러옵니다.")
            local_video = download_video(drive, selected["id"], selected["name"])
            video_hash = compute_file_hash(local_video)

            st.write("2) 입력 영상 품질을 점검합니다.")
            quality = probe_video(local_video)

            st.write("3) 이벤트 후보 구간을 추출합니다.")
            candidates = detect_event_candidates(local_video)
            if not candidates:
                status.update(label="이벤트 후보 추출 실패", state="error")
                st.error("이 영상에서 분석할 이벤트 후보 구간을 찾지 못했습니다.")
                return

            st.write("4) 후보 구간별 근거 프레임을 추출합니다.")
            frame_map = collect_candidate_frames(local_video, candidates)

            st.write("4-1) 리포트용 10초 클립을 경량 생성합니다.")
            clip_map = collect_candidate_clips(local_video, candidates)

            st.write("5) Gemini 3.5 Flash로 코칭 분석을 수행합니다.")
            try:
                result, usage = gemini_analyze(
                    api_key=gemini_api_key,
                    role=role,
                    hero=hero,
                    quality=quality,
                    candidates=candidates,
                    frame_map=frame_map,
                    video_hash=video_hash,
                )
            except Exception as exc:
                status.update(label="Gemini 분석 실패", state="error")
                st.exception(exc)
                return

            result["input_video"] = {
                "id": selected["id"],
                "name": selected["name"],
                "role": role,
                "hero": hero,
                "quality": {
                    "grade": quality.grade,
                    "width": quality.width,
                    "height": quality.height,
                    "fps": quality.fps,
                    "duration_sec": quality.duration_sec,
                    "notes": quality.notes,
                },
                "event_candidates": [candidate.__dict__ for candidate in candidates],
                "usage": usage,
                "estimated_cost_krw": estimate_cost_krw(usage),
            }
            result["result_link"] = ""
            result["upload_error"] = "manual_upload_mode"

            st.write("6) 결과 JSON을 다운로드해 직접 저장합니다.")
            status.update(label="분석 완료", state="complete")

        render_report(result, quality, candidates, frame_map, clip_map)
        return

    if not saved_reports:
        st.info("결과 폴더에 저장된 분석 JSON이 없습니다.")
        return

    selected_report = st.selectbox("불러올 분석 결과 선택", options=saved_reports, format_func=lambda item: item["name"])
    report = download_json_file(drive, selected_report["id"])
    input_video = report.get("input_video", {})
    source_video = video_by_id.get(input_video.get("id")) or video_by_name.get(input_video.get("name", ""))
    if source_video is None:
        st.error("원본 영상을 입력 폴더에서 찾지 못했습니다. 저장된 JSON만으로는 영상 재생을 복원할 수 없습니다.")
        st.download_button(
            "저장된 JSON 다운로드",
            data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=selected_report["name"],
            mime="application/json",
        )
        return

    local_video = download_video(drive, source_video["id"], source_video["name"])
    quality_data = input_video.get("quality", {})
    quality = VideoQuality(
        width=int(quality_data.get("width", 0)),
        height=int(quality_data.get("height", 0)),
        fps=float(quality_data.get("fps", 0.0)),
        duration_sec=float(quality_data.get("duration_sec", 0.0)),
        grade=str(quality_data.get("grade", "C")),
        notes=list(quality_data.get("notes", [])),
    )
    candidates = build_candidates_from_result(report)
    frame_map = collect_candidate_frames(local_video, candidates) if candidates else {}
    clip_map = collect_candidate_clips(local_video, candidates) if candidates else {}

    if report.get("model_version") != MODEL_NAME:
        st.warning(f"이 결과는 `{report.get('model_version')}` 기준으로 저장되었습니다. 현재 기본 모델은 `{MODEL_NAME}` 입니다.")

    render_report(report, quality, candidates, frame_map, clip_map)
if __name__ == "__main__":
    main()





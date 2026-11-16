"""Standalone, loopback-only local demo server for the Herdwatch dashboard.

Run with: python -m uvicorn app.farm_local:app --host 127.0.0.1 --port 8000
Temporary farmer/admin passwords are generated for each process and printed
once at startup. No database or real sensor provider is used by this demo.
"""
from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.farm_monitor import HERD

security = HTTPBearer(auto_error=False)
_passwords = {"farmer": secrets.token_urlsafe(15), "admin": secrets.token_urlsafe(15)}
_sessions: dict[str, tuple[str, float]] = {}
print("Herdwatch local demo accounts (temporary; process lifetime only):", flush=True)
print(f"  farmer / {_passwords['farmer']}", flush=True)
print(f"  admin  / {_passwords['admin']}", flush=True)

app = FastAPI(title="Herdwatch Local Demo", version="1.0.0")


class LoginRequest(BaseModel):
    username: str
    password: str


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})
    session = _sessions.get(credentials.credentials)
    if session is None or session[1] <= time.time():
        _sessions.pop(credentials.credentials, None)
        raise HTTPException(401, "Invalid or expired local demo token", headers={"WWW-Authenticate": "Bearer"})
    role = session[0]
    grants = [{"dept": "farm", "sensitivity_max": "restricted"}] if role == "farmer" else [
        {"dept": "*", "sensitivity_max": "restricted"}
    ]
    return {"user_id": f"local-{role}", "role": role, "permissions": grants}


def require_farm(user: dict[str, Any], *, admin: bool = False) -> None:
    if user.get("role") not in {"farmer", "admin"} or not any(
        grant.get("dept") in {"farm", "*"} for grant in user.get("permissions", [])
    ):
        raise HTTPException(403, "Farm dashboard access denied")
    if admin and user["role"] != "admin":
        raise HTTPException(403, "Administrator role required")


@app.get("/farm", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "farm_dashboard.html")


@app.post("/auth/token", tags=["auth"])
def login(payload: LoginRequest) -> dict[str, str]:
    expected = _passwords.get(payload.username)
    if expected is None or not secrets.compare_digest(payload.password, expected):
        raise HTTPException(401, "Invalid username or password", headers={"WWW-Authenticate": "Bearer"})
    token = secrets.token_urlsafe(32)
    _sessions[token] = (payload.username, time.time() + 8 * 60 * 60)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/farm/api/overview", tags=["farm dashboard"])
def overview(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_farm(user)
    return HERD.overview()


@app.get("/farm/api/cows", tags=["farm dashboard"])
def cows(status_filter: str = "all", user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_farm(user)
    if status_filter not in {"all", "healthy", "watch", "sick"}:
        raise HTTPException(422, "status_filter must be all, healthy, watch, or sick")
    items = HERD.snapshot()
    if status_filter != "all":
        items = [cow for cow in items if cow["health_status"] == status_filter]
    return {"count": len(items), "cows": items}


@app.get("/farm/api/alerts", tags=["farm dashboard"])
def alerts(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_farm(user)
    items = HERD.alerts(include_acknowledged=user["role"] == "admin")
    return {"count": len(items), "alerts": items}


@app.get("/farm/api/live", tags=["farm dashboard"])
def live(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_farm(user)
    feed = HERD.advance()
    return {**feed, "overview": HERD.overview(), "alerts": HERD.alerts(), "demo_data": True}


@app.post("/farm/api/alerts/{alert_id}/ack", tags=["farm dashboard"])
def acknowledge(alert_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_farm(user, admin=True)
    if not HERD.acknowledge(alert_id):
        raise HTTPException(404, "Alert not found")
    return {"alert_id": alert_id, "acknowledged": True}

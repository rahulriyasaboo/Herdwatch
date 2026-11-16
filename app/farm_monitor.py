"""Local synthetic sensor feed used by the farmer/admin dashboard.

This is a UI and API demo source, not a connection to physical collars and not
the trained model from synthetic_cow_health_hybrid.py. Replace this provider
with a validated sensor/model service before operational use.
"""
from __future__ import annotations

import random
import threading
from datetime import datetime, timezone
from typing import Any


class SyntheticHerd:
    def __init__(self, seed: int = 42, cow_count: int = 150) -> None:
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._cows: dict[int, dict[str, Any]] = {}
        self._acknowledged: set[str] = set()
        self._version = 0
        now = datetime.now(timezone.utc).isoformat()
        for cow_id in range(1, cow_count + 1):
            sick = self._rng.random() < 0.12
            self._cows[cow_id] = self._make_cow(cow_id, sick, now)

    def _make_cow(self, cow_id: int, sick: bool, timestamp: str) -> dict[str, Any]:
        temp = self._rng.gauss(39.9 if sick else 38.9, 0.22)
        pulse = self._rng.gauss(91 if sick else 65, 7)
        rumination = self._rng.gauss(-7 if sick else -0.5, 2.5)
        gait = max(0.0, min(1.0, self._rng.gauss(0.39 if sick else 0.11, 0.09)))
        thi = self._rng.uniform(60, 85)
        ventilation = self._rng.random()
        forage = self._rng.uniform(0.35, 1.0)
        activity = max(1, min(3, round(self._rng.gauss(1.5 if sick else 2.3, 0.45))))
        # Transparent demo heuristic for status visualization; not model output.
        risk = 0.08 + max(0, temp - 39.2) * 0.20 + max(0, pulse - 80) * 0.004
        risk += max(0, -rumination - 3) * 0.022 + max(0, gait - 0.2) * 0.55
        risk += max(0, thi - 75) * 0.003
        risk = min(0.99, max(0.02, risk + self._rng.uniform(-0.035, 0.035)))
        status = "sick" if risk >= 0.52 else ("watch" if risk >= 0.30 else "healthy")
        return {
            "cow_id": cow_id, "avg_temp": round(temp, 2), "avg_pulse": round(pulse, 1),
            "avg_activity": activity, "rumination_slope_6h": round(rumination, 2),
            "gait_asymmetry": round(gait, 3), "THI": round(thi, 1),
            "ventilation_index": round(ventilation, 2), "forage_index": round(forage, 2),
            "risk_score": round(risk, 3), "health_status": status,
            "updated_at": timestamp,
        }

    def advance(self) -> dict[str, Any]:
        """Generate a small batch of fresh simulated readings per poll."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            ids = self._rng.sample(list(self._cows), k=min(18, len(self._cows)))
            for cow_id in ids:
                previous = self._cows[cow_id]
                sick = previous["health_status"] == "sick" or self._rng.random() < 0.035
                if previous["health_status"] == "sick" and self._rng.random() < 0.18:
                    sick = False
                self._cows[cow_id] = self._make_cow(cow_id, sick, now)
            self._version += 1
            return {"version": self._version, "updated_at": now}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(cow) for cow in self._cows.values()]

    def alerts(self, *, include_acknowledged: bool = False) -> list[dict[str, Any]]:
        alerts = []
        for cow in self.snapshot():
            if cow["health_status"] not in {"sick", "watch"}:
                continue
            alert_id = f"cow-{cow['cow_id']}"
            acknowledged = alert_id in self._acknowledged
            if acknowledged and not include_acknowledged:
                continue
            alerts.append({
                "alert_id": alert_id, "cow_id": cow["cow_id"],
                "severity": "high" if cow["health_status"] == "sick" else "watch",
                "risk_score": cow["risk_score"], "message": _alert_message(cow),
                "updated_at": cow["updated_at"], "acknowledged": acknowledged,
            })
        return sorted(alerts, key=lambda item: (item["acknowledged"], -item["risk_score"]))

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock:
            try:
                cow_id = int(alert_id.removeprefix("cow-"))
            except ValueError:
                return False
            if cow_id not in self._cows:
                return False
            self._acknowledged.add(f"cow-{cow_id}")
            return True

    def overview(self) -> dict[str, Any]:
        cows = self.snapshot()
        alerts = self.alerts()
        healthy = sum(cow["health_status"] == "healthy" for cow in cows)
        watch = sum(cow["health_status"] == "watch" for cow in cows)
        sick = sum(cow["health_status"] == "sick" for cow in cows)
        return {
            "herd_size": len(cows), "healthy": healthy, "watch": watch, "sick": sick,
            "active_alerts": len(alerts),
            "average_risk": round(sum(cow["risk_score"] for cow in cows) / max(1, len(cows)), 3),
            "average_thi": round(sum(cow["THI"] for cow in cows) / max(1, len(cows)), 1),
            "updated_at": datetime.now(timezone.utc).isoformat(), "demo_data": True,
        }


def _alert_message(cow: dict[str, Any]) -> str:
    signs = []
    if cow["avg_temp"] > 39.5:
        signs.append("elevated temperature")
    if cow["avg_pulse"] > 80:
        signs.append("high pulse")
    if cow["rumination_slope_6h"] < -5:
        signs.append("rumination drop")
    if cow["gait_asymmetry"] > 0.3:
        signs.append("gait asymmetry")
    return ", ".join(signs) or "health indicators need review"


HERD = SyntheticHerd()

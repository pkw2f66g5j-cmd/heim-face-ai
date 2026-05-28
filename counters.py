import os
import json
from datetime import date

COUNTERS_FILE = "counters.json"


def _load() -> dict:
    if os.path.exists(COUNTERS_FILE):
        try:
            with open(COUNTERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total": 0, "today_date": str(date.today()), "today": 0, "users": {}}


def _save(data: dict):
    with open(COUNTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def increment_counter(user_id: int):
    """Записать один завершённый разбор: общий, дневной и пользовательский."""
    c = _load()
    today = str(date.today())
    if c.get("today_date") != today:
        c["today_date"] = today
        c["today"] = 0
    c["total"] = c.get("total", 0) + 1
    c["today"] = c.get("today", 0) + 1
    uid = str(user_id)
    c["users"][uid] = c["users"].get(uid, 0) + 1
    _save(c)


def get_user_count(user_id: int) -> int:
    return _load()["users"].get(str(user_id), 0)


def get_today_count() -> int:
    c = _load()
    if c.get("today_date") != str(date.today()):
        return 0
    return c.get("today", 0)


def get_total_count() -> int:
    return _load().get("total", 0)

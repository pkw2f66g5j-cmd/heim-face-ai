import os
import json
import uuid
import threading
from datetime import date, datetime

from config import PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN

COUNTERS_FILE = "counters.json"
PRODUCTS = {PRODUCT_FACE_REPORT, PRODUCT_PREMIUM_PLAN}
ORDER_PENDING = "pending"
ORDER_PAID = "paid"
ORDER_CONSUMED = "consumed"
ORDER_FAILED = "failed"
_LOCK = threading.RLock()


def _default_data() -> dict:
    return {
        "total": 0,
        "today_date": str(date.today()),
        "today": 0,
        "face_report_count": 0,
        "premium_plan_count": 0,
        "users": {},
        "selected_products": {},
        "orders": {},
        "paid_orders_count": 0,
        "paid_face_report_count": 0,
        "paid_premium_plan_count": 0,
        "credits": {
            "users": {},
            "ledger": [],
        },
        "referrals": {},
        "user_results": {},
        "return_tasks": [],
    }


def _normalize(data: dict) -> dict:
    if not isinstance(data, dict):
        data = {}

    defaults = _default_data()
    for key, value in defaults.items():
        if key not in data:
            data[key] = value

    if not isinstance(data.get("users"), dict):
        data["users"] = {}
    if not isinstance(data.get("selected_products"), dict):
        data["selected_products"] = {}
    if not isinstance(data.get("orders"), dict):
        data["orders"] = {}
    if not isinstance(data.get("credits"), dict):
        data["credits"] = defaults["credits"]

    credits = data["credits"]
    if not isinstance(credits.get("users"), dict):
        credits["users"] = {}
    if not isinstance(credits.get("ledger"), list):
        credits["ledger"] = []

    if not isinstance(data.get("referrals"), dict):
        data["referrals"] = {}

    if not isinstance(data.get("user_results"), dict):
        data["user_results"] = {}

    if not isinstance(data.get("return_tasks"), list):
        data["return_tasks"] = []

    return data


def _load() -> dict:
    with _LOCK:
        if os.path.exists(COUNTERS_FILE):
            try:
                with open(COUNTERS_FILE, "r", encoding="utf-8") as f:
                    return _normalize(json.load(f))
            except Exception:
                pass
        return _default_data()


def _save(data: dict):
    with _LOCK:
        with open(COUNTERS_FILE, "w", encoding="utf-8") as f:
            json.dump(_normalize(data), f, ensure_ascii=False, indent=2)


def increment_counter(user_id: int, product: str | None = None):
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

    if product == PRODUCT_PREMIUM_PLAN:
        c["premium_plan_count"] = c.get("premium_plan_count", 0) + 1
    elif product == PRODUCT_FACE_REPORT:
        c["face_report_count"] = c.get("face_report_count", 0) + 1

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


def set_selected_product(user_id: int, product: str):
    """Сохранить выбранный тариф пользователя до завершения сценария."""
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product: {product}")
    c = _load()
    c["selected_products"][str(user_id)] = product
    _save(c)


def get_selected_product(user_id: int) -> str | None:
    product = _load()["selected_products"].get(str(user_id))
    return product if product in PRODUCTS else None


def clear_selected_product(user_id: int):
    c = _load()
    c["selected_products"].pop(str(user_id), None)
    _save(c)


def get_product_counts() -> dict:
    c = _load()
    return {
        "face_report_count": c.get("face_report_count", 0),
        "premium_plan_count": c.get("premium_plan_count", 0),
    }


def get_user_credits(user_id: int) -> int:
    """Заготовка под будущую кредитную модель оплат/пакетов."""
    uid = str(user_id)
    user_credits = _load()["credits"]["users"].get(uid, {})
    return int(user_credits.get("balance", 0))


def add_user_credits(user_id: int, amount: int, reason: str = "manual"):
    if amount <= 0:
        return
    c = _load()
    uid = str(user_id)
    users = c["credits"]["users"]
    users.setdefault(uid, {"balance": 0})
    users[uid]["balance"] = int(users[uid].get("balance", 0)) + amount
    c["credits"]["ledger"].append({
        "user_id": uid,
        "delta": amount,
        "reason": reason,
        "date": str(date.today()),
    })
    _save(c)


def consume_user_credit(user_id: int, reason: str = "analysis") -> bool:
    c = _load()
    uid = str(user_id)
    users = c["credits"]["users"]
    users.setdefault(uid, {"balance": 0})
    balance = int(users[uid].get("balance", 0))
    if balance <= 0:
        return False
    users[uid]["balance"] = balance - 1
    c["credits"]["ledger"].append({
        "user_id": uid,
        "delta": -1,
        "reason": reason,
        "date": str(date.today()),
    })
    _save(c)
    return True


def create_order(user_id: int, product: str, provider: str, amount: int, currency: str,
                 payment_method: str | None = None) -> dict:
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product: {product}")
    order_id = uuid.uuid4().hex
    order = {
        "order_id": order_id,
        "user_id": str(user_id),
        "product": product,
        "provider": provider,
        "payment_method": payment_method,
        "amount": amount,
        "currency": currency,
        "status": ORDER_PENDING,
        "provider_payment_id": None,
        "confirmation_url": None,
        "created_date": str(date.today()),
        "paid_date": None,
        "consumed_date": None,
    }
    c = _load()
    c["orders"][order_id] = order
    _save(c)
    return order


def get_order(order_id: str) -> dict | None:
    return _load()["orders"].get(order_id)


def find_order_by_provider_payment_id(provider_payment_id: str) -> dict | None:
    for order in _load()["orders"].values():
        if order.get("provider_payment_id") == provider_payment_id:
            return order
    return None


def update_order_payment(order_id: str, provider_payment_id: str | None = None,
                         confirmation_url: str | None = None, status: str | None = None) -> dict | None:
    c = _load()
    order = c["orders"].get(order_id)
    if not order:
        return None
    if provider_payment_id:
        order["provider_payment_id"] = provider_payment_id
    if confirmation_url:
        order["confirmation_url"] = confirmation_url
    if status:
        order["provider_status"] = status
    _save(c)
    return order


def mark_order_paid(order_id: str, provider_payment_id: str | None = None) -> dict | None:
    c = _load()
    order = c["orders"].get(order_id)
    if not order:
        return None
    already_paid = order.get("status") in {ORDER_PAID, ORDER_CONSUMED}
    if provider_payment_id:
        order["provider_payment_id"] = provider_payment_id
    if not already_paid:
        order["status"] = ORDER_PAID
        order["paid_date"] = str(date.today())
        c["paid_orders_count"] = c.get("paid_orders_count", 0) + 1
        if order["product"] == PRODUCT_PREMIUM_PLAN:
            c["paid_premium_plan_count"] = c.get("paid_premium_plan_count", 0) + 1
        elif order["product"] == PRODUCT_FACE_REPORT:
            c["paid_face_report_count"] = c.get("paid_face_report_count", 0) + 1
        c["selected_products"][order["user_id"]] = order["product"]
    _save(c)
    return order


def mark_order_failed(order_id: str) -> dict | None:
    c = _load()
    order = c["orders"].get(order_id)
    if not order:
        return None
    if order.get("status") == ORDER_PENDING:
        order["status"] = ORDER_FAILED
    _save(c)
    return order


def get_active_paid_order(user_id: int) -> dict | None:
    uid = str(user_id)
    orders = [
        order for order in _load()["orders"].values()
        if order.get("user_id") == uid and order.get("status") == ORDER_PAID
    ]
    orders.sort(key=lambda x: x.get("paid_date") or x.get("created_date") or "")
    return orders[0] if orders else None


def consume_paid_order(order_id: str) -> dict | None:
    c = _load()
    order = c["orders"].get(order_id)
    if not order or order.get("status") != ORDER_PAID:
        return None
    order["status"] = ORDER_CONSUMED
    order["consumed_date"] = str(date.today())
    c["selected_products"].pop(order["user_id"], None)
    _save(c)
    return order


# ================== REFERRALS ==================
REFERRAL_GOAL = 3


def _default_ref_record() -> dict:
    return {
        "referred_by": None,        # кто пригласил (user_id строкой)
        "paid_referrals_count": 0,  # сколько приглашённых оплатили
        "free_premium_count": 0,    # доступных бесплатных Premium
        "credited_referrals": [],   # user_id приглашённых, уже засчитанных (без дублей)
    }


def _ref_record(c: dict, user_id) -> dict:
    uid = str(user_id)
    rec = c["referrals"].get(uid)
    if not isinstance(rec, dict):
        rec = _default_ref_record()
        c["referrals"][uid] = rec
    else:
        for k, v in _default_ref_record().items():
            rec.setdefault(k, v)
    return rec


def ensure_user(user_id: int):
    """Гарантирует наличие реферальной записи для пользователя."""
    c = _load()
    _ref_record(c, user_id)
    _save(c)


def set_referrer(user_id: int, referrer_id: int) -> bool:
    """Привязывает реферера к пользователю. Возвращает True, если привязка применена.
    Не перезаписывает существующего реферера и запрещает самореферал."""
    if int(user_id) == int(referrer_id):
        return False
    c = _load()
    rec = _ref_record(c, user_id)
    if rec["referred_by"]:           # уже привязан — не перезаписываем
        _save(c)
        return False
    _ref_record(c, referrer_id)      # убедимся, что у реферера тоже есть запись
    rec["referred_by"] = str(referrer_id)
    _save(c)
    return True


def get_referrer(user_id: int) -> str | None:
    return _load()["referrals"].get(str(user_id), {}).get("referred_by")


def get_referral_stats(user_id: int) -> dict:
    rec = _load()["referrals"].get(str(user_id)) or _default_ref_record()
    return {
        "paid_referrals_count": rec.get("paid_referrals_count", 0),
        "free_premium_count": rec.get("free_premium_count", 0),
        "goal": REFERRAL_GOAL,
    }


def credit_referral(referee_id: int) -> dict | None:
    """Засчитывает оплату приглашённого пользователя его рефереру.

    Идемпотентно: один referee засчитывается рефереру только один раз.
    Возвращает dict с данными для уведомления реферера или None,
    если реферера нет / уже засчитан / самореферал.
    Если достигнут порог — начисляет бесплатный Premium.
    """
    c = _load()
    referee = _ref_record(c, referee_id)
    referrer_id = referee.get("referred_by")
    if not referrer_id:
        _save(c)
        return None
    if str(referrer_id) == str(referee_id):
        _save(c)
        return None

    referrer = _ref_record(c, referrer_id)
    if str(referee_id) in referrer["credited_referrals"]:
        _save(c)
        return None  # уже засчитан, без дублей

    referrer["credited_referrals"].append(str(referee_id))
    referrer["paid_referrals_count"] = len(referrer["credited_referrals"])

    granted_premium = False
    # На каждые REFERRAL_GOAL оплаченных рефералов — один бесплатный Premium.
    earned = referrer["paid_referrals_count"] // REFERRAL_GOAL
    already = referrer.get("_granted_premium_total", 0)
    if earned > already:
        referrer["free_premium_count"] += (earned - already)
        referrer["_granted_premium_total"] = earned
        granted_premium = True

    _save(c)
    return {
        "referrer_id": str(referrer_id),
        "paid_referrals_count": referrer["paid_referrals_count"],
        "goal": REFERRAL_GOAL,
        "granted_premium": granted_premium,
    }


def has_free_premium(user_id: int) -> bool:
    return _load()["referrals"].get(str(user_id), {}).get("free_premium_count", 0) > 0


def consume_free_premium(user_id: int) -> bool:
    """Списывает один бесплатный Premium. Возвращает True при успехе."""
    c = _load()
    rec = _ref_record(c, user_id)
    if rec.get("free_premium_count", 0) <= 0:
        _save(c)
        return False
    rec["free_premium_count"] -= 1
    _save(c)
    return True


def mark_order_referral_credited(order_id: str) -> bool:
    """Помечает заказ как учтённый в реферальной программе (защита от дублей)."""
    c = _load()
    order = c["orders"].get(order_id)
    if not order:
        return False
    order["referral_credited"] = True
    _save(c)
    return True


# ================== FEEDBACK / RATINGS ==================
def add_rating(user_id: int, rating: int):
    """Сохраняет оценку качества разбора (1–5). Просто и без отдельной БД."""
    c = _load()
    if "ratings" not in c or not isinstance(c.get("ratings"), dict):
        c["ratings"] = {"list": [], "sum": 0, "count": 0}
    r = c["ratings"]
    r.setdefault("list", [])
    r["list"].append({"user_id": str(user_id), "rating": int(rating), "date": str(date.today())})
    r["sum"] = r.get("sum", 0) + int(rating)
    r["count"] = r.get("count", 0) + 1
    _save(c)


def get_ratings_summary() -> dict:
    r = _load().get("ratings") or {}
    count = r.get("count", 0)
    avg = (r.get("sum", 0) / count) if count else 0
    return {"count": count, "avg": round(avg, 2)}


# ================== USER RESULT SNAPSHOTS ==================
# Снимки результатов разборов для будущего сравнения и динамики.
# Хранятся в data["user_results"][str(user_id)] = [ {snapshot}, ... ]
# в хронологическом порядке (последний — самый свежий).

def _snapshot_signature(snap: dict) -> tuple:
    """Подпись снимка для защиты от дублей при повторной отправке PDF."""
    return (
        round(float(snap.get("score") or 0), 2),
        str(snap.get("tier") or ""),
        str(snap.get("product") or ""),
        str(snap.get("weakest_metric") or ""),
    )


def save_result_snapshot(user_id: int, score=None, tier=None, top_percent=None,
                         weakest_metric=None, product=None, paid_date=None) -> bool:
    """Сохраняет снимок результата пользователя.

    Защита:
      - не сохраняет пустые результаты (нет score или tier);
      - не сохраняет дубликат, если последний снимок идентичен по
        (score, tier, product, weakest_metric) и создан в тот же день.
    Возвращает True, если снимок добавлен, иначе False.
    Никогда не бросает исключение наружу при некорректных данных.
    """
    try:
        if score is None or tier is None:
            return False
        try:
            score_val = round(float(score), 2)
        except (TypeError, ValueError):
            return False

        snap = {
            "score": score_val,
            "tier": str(tier),
            "top_percent": top_percent,
            "weakest_metric": str(weakest_metric) if weakest_metric is not None else None,
            "product": str(product) if product is not None else None,
            "paid_date": str(paid_date) if paid_date is not None else None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        c = _load()
        uid = str(user_id)
        history = c["user_results"].get(uid)
        if not isinstance(history, list):
            history = []
            c["user_results"][uid] = history

        if history:
            last = history[-1]
            same = _snapshot_signature(last) == _snapshot_signature(snap)
            same_day = str(last.get("created_at", ""))[:10] == snap["created_at"][:10]
            if same and same_day:
                return False  # дубль повторной отправки — не сохраняем

        history.append(snap)
        _save(c)
        return True
    except Exception:
        # Сохранение снимка не должно влиять на выдачу разбора.
        return False


def get_last_snapshot(user_id: int) -> dict | None:
    """Последний (самый свежий) снимок результата пользователя или None."""
    history = _load()["user_results"].get(str(user_id))
    if isinstance(history, list) and history:
        return history[-1]
    return None


def get_previous_snapshot(user_id: int) -> dict | None:
    """Предпоследний снимок (для сравнения «было → стало») или None."""
    history = _load()["user_results"].get(str(user_id))
    if isinstance(history, list) and len(history) >= 2:
        return history[-2]
    return None


def get_result_history(user_id: int) -> list:
    """Полная история снимков пользователя (хронологически)."""
    history = _load()["user_results"].get(str(user_id))
    return list(history) if isinstance(history, list) else []


# ================== RETURN TASKS (post-purchase 7/30/60) ==================
# Персистентная очередь возврата. data["return_tasks"] = [ {task}, ... ]
# task: {id, user_id, type(day_7|day_30|day_60), product, due_at(ISO),
#        sent(bool), created_at(ISO), sent_at(ISO|None)}

_RETURN_OFFSETS = {"day_7": 7, "day_30": 30, "day_60": 60}


def _task_dedup_key(user_id, product, due_date_str, ttype) -> tuple:
    return (str(user_id), str(product or ""), str(due_date_str), str(ttype))


def enqueue_return_tasks(user_id: int, product: str, base_date=None) -> int:
    """Создаёт задачи возврата на +7/+30/+60 дней от base_date (по умолчанию now).
    Дедуп: не создаёт задачу с тем же user_id+product+date+type.
    Возвращает число реально добавленных задач. Не бросает исключений."""
    try:
        from datetime import timedelta
        base = base_date or datetime.now()
        if isinstance(base, str):
            try:
                base = datetime.fromisoformat(base)
            except ValueError:
                base = datetime.now()

        c = _load()
        tasks = c["return_tasks"]
        existing = {
            _task_dedup_key(t.get("user_id"), t.get("product"),
                            str(t.get("due_at", ""))[:10], t.get("type"))
            for t in tasks if isinstance(t, dict)
        }
        added = 0
        for ttype, days in _RETURN_OFFSETS.items():
            due = base + timedelta(days=days)
            key = _task_dedup_key(user_id, product, due.isoformat()[:10], ttype)
            if key in existing:
                continue
            tasks.append({
                "id": uuid.uuid4().hex,
                "user_id": int(user_id),
                "type": ttype,
                "product": str(product) if product is not None else None,
                "due_at": due.isoformat(timespec="seconds"),
                "sent": False,
                "created_at": base.isoformat(timespec="seconds"),
                "sent_at": None,
            })
            existing.add(key)
            added += 1
        if added:
            _save(c)
        return added
    except Exception:
        return 0


def get_due_return_tasks(now=None) -> list:
    """Возвращает неотправленные задачи с due_at <= now (хронологически)."""
    try:
        now = now or datetime.now()
        if isinstance(now, str):
            now = datetime.fromisoformat(now)
        tasks = _load()["return_tasks"]
        due = []
        for t in tasks:
            if not isinstance(t, dict) or t.get("sent"):
                continue
            try:
                if datetime.fromisoformat(t["due_at"]) <= now:
                    due.append(t)
            except (KeyError, ValueError, TypeError):
                continue
        due.sort(key=lambda t: t.get("due_at", ""))
        return due
    except Exception:
        return []


def mark_return_task_sent(task_id: str) -> bool:
    """Помечает задачу как отправленную (sent=true, sent_at=now)."""
    try:
        c = _load()
        for t in c["return_tasks"]:
            if isinstance(t, dict) and t.get("id") == task_id:
                t["sent"] = True
                t["sent_at"] = datetime.now().isoformat(timespec="seconds")
                _save(c)
                return True
        return False
    except Exception:
        return False


def cleanup_old_return_tasks(keep_days: int = 90) -> int:
    """Удаляет отправленные задачи старше keep_days. Возвращает число удалённых."""
    try:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=keep_days)
        c = _load()
        before = len(c["return_tasks"])
        kept = []
        for t in c["return_tasks"]:
            if not isinstance(t, dict):
                continue
            if t.get("sent") and t.get("sent_at"):
                try:
                    if datetime.fromisoformat(t["sent_at"]) < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            kept.append(t)
        removed = before - len(kept)
        if removed:
            c["return_tasks"] = kept
            _save(c)
        return removed
    except Exception:
        return 0


def was_return_sent_today(user_id: int, now=None) -> bool:
    """Антиспам: была ли пользователю отправлена return-задача сегодня."""
    try:
        now = now or datetime.now()
        if isinstance(now, str):
            now = datetime.fromisoformat(now)
        today = now.isoformat()[:10]
        for t in _load()["return_tasks"]:
            if (isinstance(t, dict) and t.get("sent")
                    and int(t.get("user_id", -1)) == int(user_id)
                    and str(t.get("sent_at", ""))[:10] == today):
                return True
        return False
    except Exception:
        return False

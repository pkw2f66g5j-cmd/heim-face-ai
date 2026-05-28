import os
import json
import uuid
import threading
from datetime import date

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

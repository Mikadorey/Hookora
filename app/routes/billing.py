import hashlib
import hmac
import json
import secrets
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.utils.security import get_current_user

router = APIRouter(
    prefix="/billing",
    tags=["Billing"],
)

PAYSTACK_BASE_URL = "https://api.paystack.co"


class CreateCheckoutSessionRequest(BaseModel):
    plan: str = Field(..., pattern="^(creator|pro)$")


ACTIVE_BILLING_STATUSES = {"active", "trialing", "past_due", "unpaid"}


def _to_plain_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return {}


def _paystack_headers() -> dict:
    secret_key = settings.paystack_secret_key.strip()

    if not secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paystack secret key is not configured.",
        )

    return {
        "Authorization": f"Bearer {secret_key}",
        "Content-Type": "application/json",
    }


def _get_plan_code(plan: str) -> str:
    normalized = (plan or "").strip().lower()

    if normalized == "creator":
        return settings.paystack_plan_creator_monthly.strip()

    if normalized == "pro":
        return settings.paystack_plan_pro_monthly.strip()

    return ""


def _get_plan_amount(plan: str) -> int:
    normalized = (plan or "").strip().lower()

    if normalized == "creator":
        return 1200000  # ₦12,000 in kobo

    if normalized == "pro":
        return 2400000  # ₦24,000 in kobo

    return 0


def _get_plan_name_from_plan_code(plan_code: str | None) -> str:
    normalized = (plan_code or "").strip()

    if normalized and normalized == settings.paystack_plan_creator_monthly.strip():
        return "creator"

    if normalized and normalized == settings.paystack_plan_pro_monthly.strip():
        return "pro"

    return "free"


def _to_naive_datetime(value: str | None):
    if not value:
        return None

    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).replace(tzinfo=None)
    except Exception:
        return None


def _build_reference(user_id: int, plan: str) -> str:
    token = secrets.token_hex(8)
    return f"hookora_{plan}_{user_id}_{token}"


def _extract_customer_code(data: dict) -> str | None:
    customer = data.get("customer")

    if isinstance(customer, dict):
        return customer.get("customer_code") or customer.get("code")

    if isinstance(customer, str):
        return customer

    return None


def _extract_subscription_code(data: dict) -> str | None:
    subscription = data.get("subscription")

    if isinstance(subscription, dict):
        return subscription.get("subscription_code") or subscription.get("code")

    if isinstance(subscription, str):
        return subscription

    return data.get("subscription_code")


def _extract_plan_code(data: dict) -> str | None:
    plan = data.get("plan")

    if isinstance(plan, dict):
        return plan.get("plan_code") or plan.get("code")

    if isinstance(plan, str):
        return plan

    return data.get("plan_code")


def _find_user_for_event(db: Session, data: dict) -> User | None:
    metadata = _to_plain_dict(data.get("metadata"))
    user_id = metadata.get("user_id")

    if user_id:
        try:
            user = db.query(User).filter(User.id == int(user_id)).first()
            if user:
                return user
        except Exception:
            pass

    customer_code = _extract_customer_code(data)
    if customer_code:
        user = db.query(User).filter(User.paystack_customer_code == customer_code).first()
        if user:
            return user

    subscription_code = _extract_subscription_code(data)
    if subscription_code:
        user = db.query(User).filter(User.paystack_subscription_code == subscription_code).first()
        if user:
            return user

    customer_email = None
    customer = data.get("customer")
    if isinstance(customer, dict):
        customer_email = customer.get("email")

    if customer_email:
        user = db.query(User).filter(User.email == customer_email).first()
        if user:
            return user

    return None


def _sync_user_from_paystack_data(
    db: Session,
    user: User,
    data: dict,
    *,
    fallback_plan: str | None = None,
    billing_status: str | None = None,
):
    customer_code = _extract_customer_code(data)
    subscription_code = _extract_subscription_code(data)
    plan_code = _extract_plan_code(data)

    if customer_code:
        user.paystack_customer_code = customer_code

    if subscription_code:
        user.paystack_subscription_code = subscription_code

    if plan_code:
        user.paystack_plan_code = plan_code

    resolved_plan = _get_plan_name_from_plan_code(plan_code)
    if resolved_plan == "free" and fallback_plan in {"creator", "pro"}:
        resolved_plan = fallback_plan

    user.plan = resolved_plan if resolved_plan in {"creator", "pro"} else "free"

    if billing_status:
        user.billing_status = billing_status
    else:
        user.billing_status = data.get("status") or user.billing_status or "active"

    next_payment_date = data.get("next_payment_date")
    paid_at = data.get("paid_at")
    user.billing_current_period_end = (
        _to_naive_datetime(next_payment_date) or _to_naive_datetime(paid_at)
    )

    db.commit()
    db.refresh(user)


@router.post("/checkout-session")
def create_checkout_session(
    payload: CreateCheckoutSessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    requested_plan = payload.plan.strip().lower()

    if (
        current_user.plan == requested_plan
        and (current_user.billing_status or "").lower() in ACTIVE_BILLING_STATUSES
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You are already on the {requested_plan} plan.",
        )

    plan_code = _get_plan_code(requested_plan)
    if not plan_code:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paystack plan code is not configured for the {requested_plan} plan.",
        )

    amount = _get_plan_amount(requested_plan)
    if amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paystack amount is not configured for the {requested_plan} plan.",
        )

    reference = _build_reference(current_user.id, requested_plan)

    callback_url = (
        f"{settings.frontend_app_url.rstrip('/')}/app/pricing"
        f"?checkout=success&provider=paystack&reference={reference}"
    )
    cancel_url = (
        f"{settings.frontend_app_url.rstrip('/')}/app/pricing"
        f"?checkout=cancelled&provider=paystack"
    )

    body = {
        "email": current_user.email,
        "amount": amount,
        "plan": plan_code,
        "reference": reference,
        "callback_url": callback_url,
        "metadata": {
            "user_id": str(current_user.id),
            "requested_plan": requested_plan,
            "cancel_action": cancel_url,
        },
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{PAYSTACK_BASE_URL}/transaction/initialize",
                headers=_paystack_headers(),
                json=body,
            )

        try:
            data = response.json()
        except Exception:
            data = {}

        if response.status_code >= 400 or not data.get("status"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=data.get("message") or "Failed to initialize Paystack checkout.",
            )

        checkout_data = _to_plain_dict(data.get("data"))
        checkout_url = checkout_data.get("authorization_url")

        if not checkout_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Paystack did not return a checkout URL.",
            )

        return {
            "checkout_url": checkout_url,
            "reference": checkout_data.get("reference"),
            "access_code": checkout_data.get("access_code"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.get("/status")
def get_billing_status(
    current_user: User = Depends(get_current_user),
):
    return {
        "plan": current_user.plan,
        "billing_status": current_user.billing_status,
        "paystack_customer_code": current_user.paystack_customer_code,
        "paystack_subscription_code": current_user.paystack_subscription_code,
        "paystack_plan_code": current_user.paystack_plan_code,
        "billing_current_period_end": (
            current_user.billing_current_period_end.isoformat()
            if current_user.billing_current_period_end
            else None
        ),
    }


@router.post("/webhook")
async def paystack_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    secret_key = settings.paystack_secret_key.strip()

    if not secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paystack secret key is not configured.",
        )

    payload = await request.body()
    signature = request.headers.get("x-paystack-signature", "").strip()

    computed_signature = hmac.new(
        secret_key.encode("utf-8"),
        payload,
        hashlib.sha512,
    ).hexdigest()

    if not signature or not hmac.compare_digest(computed_signature, signature):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Paystack webhook signature.",
        )

    try:
        event = json.loads(payload.decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook payload.",
        )

    event_type = event.get("event")
    data = _to_plain_dict(event.get("data"))

    user = _find_user_for_event(db, data)
    if not user:
        return {"received": True}

    metadata = _to_plain_dict(data.get("metadata"))
    requested_plan = metadata.get("requested_plan")

    if event_type in {"charge.success", "invoice.create", "invoice.update"}:
        _sync_user_from_paystack_data(
            db,
            user,
            data,
            fallback_plan=requested_plan,
            billing_status="active",
        )

    elif event_type == "subscription.create":
        _sync_user_from_paystack_data(
            db,
            user,
            data,
            fallback_plan=requested_plan or _get_plan_name_from_plan_code(_extract_plan_code(data)),
            billing_status="active",
        )

    elif event_type == "subscription.disable":
        user.billing_status = "inactive"
        user.plan = "free"
        user.paystack_subscription_code = None
        user.paystack_plan_code = None
        user.billing_current_period_end = None
        db.commit()
        db.refresh(user)

    elif event_type == "invoice.payment_failed":
        user.billing_status = "past_due"
        db.commit()
        db.refresh(user)

    return {"received": True}
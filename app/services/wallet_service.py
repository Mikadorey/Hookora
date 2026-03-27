from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.wallet_ledger import WalletLedger


def compute_wallet_balance(db: Session, user_id: int) -> int:
    total = (
        db.query(func.coalesce(func.sum(WalletLedger.amount_naira), 0))
        .filter(
            WalletLedger.user_id == user_id,
            WalletLedger.status == "posted",
        )
        .scalar()
    )
    return int(total or 0)


def refresh_wallet_balance(db: Session, user_id: int) -> int:
    balance = compute_wallet_balance(db, user_id)
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.wallet_balance_naira = balance
        db.commit()
        db.refresh(user)
    return balance


def post_wallet_entry(
    db: Session,
    user_id: int,
    entry_type: str,
    amount_naira: int,
    reference: str | None = None,
    metadata: dict | None = None,
) -> WalletLedger:
    if reference:
        existing = (
            db.query(WalletLedger)
            .filter(
                WalletLedger.user_id == user_id,
                WalletLedger.entry_type == entry_type,
                WalletLedger.reference == reference,
            )
            .first()
        )
        if existing:
            return existing

    entry = WalletLedger(
        user_id=user_id,
        entry_type=entry_type,
        amount_naira=amount_naira,
        status="posted",
        reference=reference,
        metadata_json=metadata or {},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    refresh_wallet_balance(db, user_id)
    return entry


def get_available_wallet_balance(db: Session, user_id: int) -> int:
    return max(0, compute_wallet_balance(db, user_id))


def preview_wallet_application(
    db: Session,
    user_id: int,
    amount_after_rewards_naira: int,
) -> dict:
    available = get_available_wallet_balance(db, user_id)
    applied = min(available, max(0, amount_after_rewards_naira))
    remaining = max(0, amount_after_rewards_naira - applied)

    return {
        "wallet_balance_available_naira": available,
        "wallet_balance_applied_naira": applied,
        "card_due_naira": remaining,
    }


def consume_wallet_balance(
    db: Session,
    user_id: int,
    amount_naira: int,
    reference: str,
    metadata: dict | None = None,
) -> bool:
    amount = int(amount_naira or 0)
    if amount <= 0:
        return False

    available = get_available_wallet_balance(db, user_id)
    if amount > available:
        amount = available

    if amount <= 0:
        return False

    post_wallet_entry(
        db=db,
        user_id=user_id,
        entry_type="billing_applied",
        amount_naira=-amount,
        reference=reference,
        metadata=metadata or {},
    )
    return True


def credit_wallet_funding(
    db: Session,
    user_id: int,
    amount_naira: int,
    reference: str,
    metadata: dict | None = None,
) -> WalletLedger:
    amount = int(amount_naira or 0)
    if amount <= 0:
        raise ValueError("Funding amount must be greater than zero.")

    return post_wallet_entry(
        db=db,
        user_id=user_id,
        entry_type="funding",
        amount_naira=amount,
        reference=reference,
        metadata=metadata or {},
    )
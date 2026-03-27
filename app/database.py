from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

SQLALCHEMY_DATABASE_URL = settings.database_url.strip()

connect_args = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _is_sqlite() -> bool:
    return SQLALCHEMY_DATABASE_URL.startswith("sqlite")


def _datetime_sql_type() -> str:
    return "DATETIME" if _is_sqlite() else "TIMESTAMP"


def _json_sql_type() -> str:
    return "JSON" if _is_sqlite() else "JSONB"


def init_db():
    from app.models.hook import Hook  # noqa: F401
    from app.models.referral import Referral  # noqa: F401
    from app.models.reward_credit_ledger import RewardCreditLedger  # noqa: F401
    from app.models.wallet_ledger import WalletLedger  # noqa: F401
    from app.models.studio_asset import StudioAsset  # noqa: F401
    from app.models.studio_usage_event import StudioUsageEvent  # noqa: F401
    from app.models.user import User  # noqa: F401
    from app.models.waitlist_interest import WaitlistInterest  # noqa: F401

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    datetime_type = _datetime_sql_type()
    json_type = _json_sql_type()

    if "hooks" in tables:
        columns = [column["name"] for column in inspector.get_columns("hooks")]

        if "is_favorite" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE hooks ADD COLUMN is_favorite BOOLEAN NOT NULL DEFAULT 0"
                    )
                )

    if "users" in tables:
        columns = [column["name"] for column in inspector.get_columns("users")]

        if "plan" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN plan VARCHAR NOT NULL DEFAULT 'free'"
                    )
                )

        if "monthly_generation_count" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN monthly_generation_count INTEGER NOT NULL DEFAULT 0"
                    )
                )

        if "generation_reset_date" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN generation_reset_date {datetime_type}"
                    )
                )

        if "created_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN created_at {datetime_type}"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                    )
                )

        if "billing_status" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN billing_status VARCHAR NOT NULL DEFAULT 'inactive'"
                    )
                )

        if "billing_current_period_end" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN billing_current_period_end {datetime_type}"
                    )
                )

        if "paystack_customer_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN paystack_customer_code VARCHAR"
                    )
                )

        if "paystack_subscription_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN paystack_subscription_code VARCHAR"
                    )
                )

        if "paystack_plan_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN paystack_plan_code VARCHAR"
                    )
                )

        if "is_email_verified" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN is_email_verified INTEGER NOT NULL DEFAULT 0"
                    )
                )

        if "email_verified_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN email_verified_at {datetime_type}"
                    )
                )

        if "referral_code" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN referral_code VARCHAR"
                    )
                )

        if "referred_by_user_id" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN referred_by_user_id INTEGER"
                    )
                )

        if "referral_source_locked_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN referral_source_locked_at {datetime_type}"
                    )
                )

        if "reward_credits_balance_naira" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN reward_credits_balance_naira INTEGER NOT NULL DEFAULT 0"
                    )
                )

        if "wallet_balance_naira" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN wallet_balance_naira INTEGER NOT NULL DEFAULT 0"
                    )
                )

        columns = [column["name"] for column in inspector.get_columns("users")]

        if "referral_code" in columns:
            with engine.begin() as connection:
                rows = connection.execute(
                    text("SELECT id FROM users WHERE referral_code IS NULL OR referral_code = ''")
                ).fetchall()

                for row in rows:
                    user_id = int(row[0])
                    referral_code = f"HKR{user_id:06d}"
                    connection.execute(
                        text(
                            "UPDATE users SET referral_code = :referral_code WHERE id = :user_id"
                        ),
                        {"referral_code": referral_code, "user_id": user_id},
                    )

    if "referrals" in tables:
        columns = [column["name"] for column in inspector.get_columns("referrals")]

        if "status" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE referrals ADD COLUMN status VARCHAR NOT NULL DEFAULT 'signed_up'"
                    )
                )

        if "rejection_reason" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE referrals ADD COLUMN rejection_reason VARCHAR"
                    )
                )

        if "signup_reward_awarded_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE referrals ADD COLUMN signup_reward_awarded_at {datetime_type}"
                    )
                )

        if "paid_reward_awarded_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE referrals ADD COLUMN paid_reward_awarded_at {datetime_type}"
                    )
                )

        if "created_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE referrals ADD COLUMN created_at {datetime_type}"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE referrals SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                    )
                )

        if "updated_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE referrals ADD COLUMN updated_at {datetime_type}"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE referrals SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL"
                    )
                )

    if "reward_credit_ledger" in tables:
        columns = [column["name"] for column in inspector.get_columns("reward_credit_ledger")]

        if "status" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE reward_credit_ledger ADD COLUMN status VARCHAR NOT NULL DEFAULT 'posted'"
                    )
                )

        if "reference" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE reward_credit_ledger ADD COLUMN reference VARCHAR"
                    )
                )

        if "metadata_json" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE reward_credit_ledger ADD COLUMN metadata_json {json_type}"
                    )
                )

        if "created_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE reward_credit_ledger ADD COLUMN created_at {datetime_type}"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE reward_credit_ledger SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                    )
                )

    if "wallet_ledger" in tables:
        columns = [column["name"] for column in inspector.get_columns("wallet_ledger")]

        if "status" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE wallet_ledger ADD COLUMN status VARCHAR NOT NULL DEFAULT 'posted'"
                    )
                )

        if "reference" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE wallet_ledger ADD COLUMN reference VARCHAR"
                    )
                )

        if "metadata_json" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE wallet_ledger ADD COLUMN metadata_json {json_type}"
                    )
                )

        if "created_at" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE wallet_ledger ADD COLUMN created_at {datetime_type}"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE wallet_ledger SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                    )
                )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
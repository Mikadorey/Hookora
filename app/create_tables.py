from app.database import Base, engine

# Import all models so SQLAlchemy sees them before create_all
from app.models.user import User  # noqa: F401
from app.models.studio_usage_event import StudioUsageEvent  # noqa: F401


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_tables()
# create_tables.py
from app.database import Base, engine
from app.models import user, hook

Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")
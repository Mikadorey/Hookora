# fresh_db.py
from app.models import Base
from sqlalchemy import create_engine

# This will create a brand-new database file
engine = create_engine("sqlite:///fresh_db.sqlite3", echo=True)

# Drop all tables just in case
Base.metadata.drop_all(bind=engine)

# Recreate tables from your current models
Base.metadata.create_all(bind=engine)

print("✅ Fresh database created successfully!")
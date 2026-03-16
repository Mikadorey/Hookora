from app.database import Base, engine

confirm = input("Type YES to reset database: ")

if confirm == "YES":
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database reset complete.")
else:
    print("Cancelled.")
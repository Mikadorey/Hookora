import firebase_admin

from firebase_admin import credentials


def initialize_firebase():
    if firebase_admin._apps:
        return

    cred = credentials.Certificate(
    "app/firebase-service-account.json"
)

    firebase_admin.initialize_app(cred)

    print("Firebase initialized successfully")
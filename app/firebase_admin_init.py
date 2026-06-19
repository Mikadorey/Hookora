import os
import json
import firebase_admin

from firebase_admin import credentials


def initialize_firebase():
    if firebase_admin._apps:
        return

    firebase_json = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT"
    )

    if not firebase_json:
        raise ValueError(
            "FIREBASE_SERVICE_ACCOUNT environment variable is missing"
        )

    cred_dict = json.loads(
        firebase_json
    )

    cred = credentials.Certificate(
        cred_dict
    )

    firebase_admin.initialize_app(
        cred
    )

    print(
        "Firebase initialized successfully"
    )
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import SessionLocal, init_db
from app.models.user import User
from app.routes.billing import router as billing_router
from app.routes.dashboard import router as dashboard_router
from app.routes.hooks import router as hooks_router
from app.routes.studio import router as studio_router
from app.routes.studio_assets import router as studio_assets_router
from app.routes.users import router as users_router
from app.routes.waitlist import router as waitlist_router


class MilestoneConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected: list[WebSocket] = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


app = FastAPI(
    title="Hookora API",
    description="AI Hook Generation and Analytics Platform",
    version="1.0",
)

app.state.milestone_manager = MilestoneConnectionManager()


@app.on_event("startup")
def on_startup():
    init_db()


allowed_origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://hookora.app",
    "https://www.hookora.app",
]

frontend_app_url = settings.frontend_app_url.strip()
if frontend_app_url and frontend_app_url not in allowed_origins:
    allowed_origins.append(frontend_app_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(hooks_router)
app.include_router(dashboard_router)
app.include_router(waitlist_router)
app.include_router(studio_assets_router)
app.include_router(studio_router)
app.include_router(billing_router)


def get_total_user_count() -> int:
    db = SessionLocal()
    try:
        return db.query(User).count()
    finally:
        db.close()


@app.websocket("/ws/milestone-watch")
async def milestone_watch_ws(websocket: WebSocket):
    manager = app.state.milestone_manager
    await manager.connect(websocket)

    try:
        await websocket.send_json({
            "type": "milestone_update",
            "total_users": get_total_user_count(),
        })

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.get("/")
def root():
    return {"message": "Welcome to Hookora API"}
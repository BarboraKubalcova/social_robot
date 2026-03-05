from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_chat import router as chat_router
# from api.routes_admin import router as admin_router
# from execution.storage.db import init_db
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Healthcare Agent API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router)
# app.include_router(admin_router)

@app.on_event("startup")
def startup_event():
    # init_db()
    # logging.info("Database initialized.")
    logging.info("App is running")

@app.get("/health")
def health_check():
    return {"status": "ok"}

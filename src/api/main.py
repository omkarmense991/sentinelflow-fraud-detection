from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.prediction import (
    router as prediction_router
)

from src.api.routes.health import (
    router as health_router
)

from src.database.init_db import (
    initialize_database
)

from src.config.settings import (
    PROJECT_NAME
)


# =========================================
# Lifespan Manager
# =========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Initializing database...")

    initialize_database()

    print("Database initialized successfully.")

    yield

    print("Shutting down API...")


# =========================================
# FastAPI Application
# =========================================

app = FastAPI(
    title=PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan
)


# =========================================
# Root Endpoint
# =========================================

@app.get("/")
def root():

    return {
        "message": PROJECT_NAME,
        "status": "running"
    }


# =========================================
# Routers
# =========================================

app.include_router(prediction_router)

app.include_router(health_router)
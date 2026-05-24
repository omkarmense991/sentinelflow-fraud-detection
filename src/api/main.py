from fastapi import FastAPI

from src.api.routes.v1.prediction import router as prediction_router

from src.api.routes.v1.health import router as health_router

from src.database.init_db import initialize_database

from contextlib import asynccontextmanager

from src.config.settings import PROJECT_NAME

from src.utils.logger import logger

import time

# =========================================
# Lifespan Manager
# =========================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Initializing database...")

    initialize_database()

    logger.info("Database initialized successfully.")

    yield

    logger.info("Shutting down API...")


# =========================================
# FastAPI Application
# =========================================

app = FastAPI(title=PROJECT_NAME, version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request, call_next):

    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    response.headers["X-Process-Time"] = str(process_time)

    return response


# =========================================
# Root Endpoint
# =========================================


@app.get("/")
def root():

    return {"message": PROJECT_NAME, "status": "running"}


# =========================================
# Routers
# =========================================
app.include_router(prediction_router, prefix="/api/v1")

app.include_router(health_router, prefix="/api/v1")

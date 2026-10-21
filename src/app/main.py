from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.ml_loader import ModelManager

# Import Module Routers
from app.modules.inference.router import router as inference_router
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # App Startup: Pre-load ML Model
    ModelManager.load_resources()
    yield
    # App Shutdown


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Register Module Routers
app.include_router(inference_router, prefix=settings.API_V1_STR)

@app.get("/")
@app.head("/")
def root():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from src.inference.v1.api.router import get_v1_router
from src.inference.v1.core.service import RecommendationService

def create_app(service: RecommendationService):
    app = FastAPI(title="Recommendation API")
    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Instrument with Prometheus
    Instrumentator().instrument(app).expose(app)
    
    # Health check
    @app.get("/health")
    def health():
        return {"status": "healthy"}
    
    # Include versioned router
    app.include_router(get_v1_router(service))
    
    return app

from fastapi import APIRouter
from src.inference.v1.api.endpoints.recommend import get_recommend_router
from src.inference.v1.api.endpoints.feedback import get_feedback_router
from src.inference.v1.api.endpoints.stats import get_stats_router
from src.inference.v1.api.endpoints.movies import get_movies_router
from src.inference.v1.api.endpoints.auth import router as auth_router
from src.inference.v1.core.service import RecommendationService

def get_v1_router(service: RecommendationService):
    router = APIRouter(prefix="/v1")

    router.include_router(auth_router, prefix="/auth", tags=["auth"])
    router.include_router(get_recommend_router(service), prefix="/recommend", tags=["recommend"])
    router.include_router(get_feedback_router(service), prefix="/feedback", tags=["feedback"])
    router.include_router(get_stats_router(service), prefix="/stats", tags=["stats"])
    router.include_router(get_movies_router(service), prefix="/movies", tags=["movies"])

    return router

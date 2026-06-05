from fastapi import APIRouter
from src.inference.v1.core.service import RecommendationService

def get_stats_router(service: RecommendationService):
    r = APIRouter()

    @r.get("")
    def get_stats():
        return {
            "num_users": len(service.user_emb),
            "num_items": len(service.item_emb),
            "cache_size": len(service.cache._store)
        }
    
    return r

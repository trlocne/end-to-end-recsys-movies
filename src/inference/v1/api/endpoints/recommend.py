from fastapi import APIRouter, HTTPException
from src.inference.v1.core.service import RecommendationService

router = APIRouter()

# We pass the service via app.state or similar, but for now we define the routes 
# and the create_app will handle the dependency injection or closure.

def get_recommend_router(service: RecommendationService):
    r = APIRouter()

    @r.get("/home")
    def recommend_home(user_id: int, num_items: int = 20):
        try:
            res = service.recommend_home(user_id, top_k=num_items)
            return res
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @r.get("/item")
    def recommend_item(item_id: int, user_id: int, num_items: int = 20):
        try:
            res = service.recommend_similar(user_id, item_id, top_k=num_items)
            return res
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    @r.get("/search")
    def recommend_search(q: str, user_id: int, num_items: int = 20):
        try:
            res = service.recommend_search(user_id, q, top_k=num_items)
            return res
        except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    
    return r

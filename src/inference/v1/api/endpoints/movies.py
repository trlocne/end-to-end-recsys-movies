from fastapi import APIRouter, HTTPException, Query

from src.inference.v1.core.service import RecommendationService


def get_movies_router(service: RecommendationService):
    r = APIRouter()

    @r.get("/batch")
    def movies_batch(ids: str = Query(..., description="Comma-separated movie_id values")):
        raw = [x.strip() for x in ids.split(",") if x.strip()]
        if len(raw) > 200:
            raise HTTPException(status_code=400, detail="Too many ids (max 200)")
        try:
            id_list = [int(x) for x in raw]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"ids must be integers: {e}") from e
        movies = service.get_movies_metadata(id_list)
        return {"movies": movies}

    return r

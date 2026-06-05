from fastapi import APIRouter, BackgroundTasks

from src.inference.v1.models.schemas import InteractionEvent
from src.inference.v1.core.service import RecommendationService
from src.inference.v1.monitoring.metrics import FEEDBACK_TOTAL


def get_feedback_router(service: RecommendationService):
    r = APIRouter()

    @r.post("/click")
    def log_click(event: InteractionEvent, background_tasks: BackgroundTasks):
        FEEDBACK_TOTAL.labels(event_type="click").inc()
        rating = float(event.rating or 5.0)
        merged = service.merge_interaction_session(event.user_id, event.item_id, rating)
        background_tasks.add_task(
            service.persist_interaction_db,
            event.user_id,
            event.item_id,
            rating,
            merged["recent_movie_ids"],
            merged["recent_ratings"],
        )
        return {"status": "success"}

    @r.get("/recent/{user_id}")
    def get_recent(user_id: int):
        return service.get_recent_interactions(user_id)

    return r

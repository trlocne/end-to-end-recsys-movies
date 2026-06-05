from fastapi import APIRouter, HTTPException
from src.inference.v1.models.schemas import RegisterRequest, RegisterResponse, LoginRequest, LoginResponse
from src.inference.v1.core.db import get_conn

router = APIRouter()

@router.post("/register", response_model=RegisterResponse, status_code=201)
def register(req: RegisterRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM public.users WHERE user_id = %s", (req.user_id,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail=f"user_id {req.user_id} already exists")

            cur.execute(
                """
                INSERT INTO public.users (user_id, metadata)
                VALUES (%s, %s)
                RETURNING user_id, created_at
                """,
                (req.user_id, req.metadata and str(req.metadata).replace("'", '"'))
            )
            row = cur.fetchone()

    return RegisterResponse(
        user_id=row["user_id"],
        created_at=row["created_at"],
        message=f"User {row['user_id']} registered successfully"
    )


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, created_at, updated_at, metadata FROM public.users WHERE user_id = %s",
                (req.user_id,)
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"user_id {req.user_id} not found")

    return LoginResponse(**row)

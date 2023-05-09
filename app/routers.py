from fastapi import APIRouter
from app.endpoints import chatbot


router = APIRouter()
api_prefix = "/api/v1"

router.include_router(chatbot.router, tags=["Chatbot"], prefix=api_prefix)

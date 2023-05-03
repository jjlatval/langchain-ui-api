import threading

from fastapi import APIRouter
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from queue import Queue
from langchain.callbacks import get_openai_callback
from lib.agents import make_agent


class Chatbot(BaseModel):
    message: str


router = APIRouter()


@router.post("/chatbots/{chatbot_id}", name="Chatbot", description="Chatbot endpoint")
async def chatbot(chatbot_id: int, body: Chatbot):
    """Chatbot endpoint"""
    payload = body.message

    def on_llm_new_token(token: str) -> None:
        data_queue.put(token)

    def on_llm_end() -> None:
        data_queue.put("CLOSE")

    def event_stream(data_queue: Queue) -> str:
        while True:
            data = data_queue.get()
            if data == "CLOSE":
                yield f"data: {data}\n\n"
                break
            yield f"data: {data}\n\n"

    def conversation_run_thread(payload: str) -> None:
        with get_openai_callback():
            agent = make_agent(chatbot_id, on_llm_new_token, on_llm_end)
            agent.run(payload)

    data_queue = Queue()
    t = threading.Thread(target=conversation_run_thread, args=(payload,))
    t.start()
    response = StreamingResponse(
        event_stream(data_queue), media_type="text/event-stream"
    )
    return response

from typing import Any
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from lib.callbacks import StreamingLLMCallbackHandler
from lib.db import supabase as supabase_client
from lib.prompts import default_prompt


def makeAgent(chatbot_id: str, on_llm_new_token: Any, on_llm_end: Any):
    messages = (
        supabase_client.table("ChatbotMessage")
        .select("*")
        .eq("chatbotId", chatbot_id)
        .order(column="createdAt", desc=True)
        .limit(size=4)
        .execute()
    )
    history = ChatMessageHistory()
    [
        history.add_ai_message(message["message"])
        if message["agent"] == "ai"
        else history.add_user_message(message["message"])
        for message in messages.data
    ]
    memory = ConversationBufferMemory(chat_memory=history)
    llm = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[StreamingLLMCallbackHandler(on_llm_new_token, on_llm_end)],
    )
    agent = LLMChain(llm=llm, memory=memory, verbose=True, prompt=default_prompt)
    return agent

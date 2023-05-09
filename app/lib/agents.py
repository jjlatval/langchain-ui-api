import requests

from typing import Any, List
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredAPIFileIOLoader
from app.lib.callbacks import StreamingLLMCallbackHandler
from app.lib.db import supabase as supabase_client
from app.lib.prompts import default_prompt


def use_memory(chatbot_id: str) -> List:
    """Initiate memory"""
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
    return memory


def use_datasource(datasource_id: str) -> Any:
    """Fetch datasource using pandas"""
    datasource = (
        supabase_client.table("Datasource")
        .select("*")
        .eq("id", datasource_id)
        .single()
        .execute()
    )
    file_url = datasource.data["url"]
    file_type = datasource.data["type"]
    file_name = f"{file_url}.{file_type}"
    datasource.data["type"]
    file_repsonse = requests.get(file_url)
    loader = UnstructuredAPIFileIOLoader(
        file=file_repsonse.content, file_filename=file_name
    )
    docs = loader.load()
    return docs


def make_datasource_agent(
    chatbot_id: str, datasource_id: str, on_llm_new_token: Any, on_llm_end: Any
):
    """Creates an Agent for Q&A of documents"""
    datasource = use_datasource(datasource_id)
    print(datasource)
    memory = use_memory(chatbot_id)
    llm = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[StreamingLLMCallbackHandler(on_llm_new_token, on_llm_end)],
    )
    agent = LLMChain(llm=llm, memory=memory, verbose=True, prompt=default_prompt)
    return agent


def make_default_agent(chatbot_id: str, on_llm_new_token: Any, on_llm_end: Any):
    """Creates a default chat agent"""
    memory = use_memory(chatbot_id)
    llm = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[StreamingLLMCallbackHandler(on_llm_new_token, on_llm_end)],
    )
    agent = LLMChain(llm=llm, memory=memory, verbose=True, prompt=default_prompt)
    return agent


def make_agent(chatbot_id: str, on_llm_new_token: Any, on_llm_end: Any):
    """Helper method for creating different type of agents"""
    chatbot = (
        supabase_client.table("Chatbot")
        .select("*")
        .eq("id", chatbot_id)
        .single()
        .execute()
    )
    datasource = chatbot.data["datasourceId"]

    if datasource:
        agent = make_datasource_agent(
            chatbot_id=chatbot_id,
            datasource_id=datasource,
            on_llm_new_token=on_llm_new_token,
            on_llm_end=on_llm_end,
        )
    else:
        agent = make_default_agent(
            chatbot_id=chatbot_id,
            on_llm_new_token=on_llm_new_token,
            on_llm_end=on_llm_end,
        )

    return agent

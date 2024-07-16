import os
import uuid
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)

load_dotenv(override=True)


# 핵심! 사용자가 메시지를 보내면 이 함수가 실행됩니다.
@cl.on_message
async def on_message(recieved_message: cl.Message):
    send_message = cl.Message(content="")

    async for chunk in LLM.astream(recieved_message.content):
        await send_message.stream_token(chunk.content)
    await send_message.send()

    # TODO: app-flight.py를 보고 flight_agent로 바꿔보세요
    # agent 실행시키기 위해 필요한 정보
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id,
        }
    }

    agent = None
    async for chunk in agent.astream(
        {"messages": ("user", recieved_message.content)},
        config,
        stream_mode="values",
    ):
        msg = chunk["messages"][-1].content
    await send_message.stream_token(msg)
    await send_message.send()

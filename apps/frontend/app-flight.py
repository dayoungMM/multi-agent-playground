import os
import random
import chainlit as cl
from chainlit.step import Step
from literalai.helper import utc_now
from dotenv import load_dotenv
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

from langchain_openai import ChatOpenAI
from flows.booking.advanced_flight_agent import builder


LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)

load_dotenv(override=True)


# profile 설정
@cl.set_chat_profiles
async def chat_profile():
    profiles = []
    profiles.append(
        cl.ChatProfile(
            name="simple",
            markdown_description="simple chat",
            icon="https://picsum.photos/{}".format(random.choice(range(200, 250))),
        )
    )

    profiles.append(
        cl.ChatProfile(
            name="flight_assistant",
            markdown_description="flight assistant",
            icon="https://picsum.photos/{}".format(random.choice(range(200, 250))),
        )
    )
    return profiles


# 채팅 시작할때 봇이 먼저 말하기
@cl.on_chat_start
async def start() -> None:
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "simple":
        await cl.Message(
            content="안녕하세요. Simple Chatbot입니다. 무엇이 궁금하신가요?"
        ).send()
    elif chat_profile == "flight_assistant":
        await cl.Message(
            content="안녕하세요. Swiss Airline 고객센터입니다. 어떻게 도와드릴까요?"
        ).send()


def set_step(name: str, content: str) -> Step:
    step = Step(name=name, type="tool")
    step.start = utc_now()
    step.input = content
    return step


# 핵심! 사용자가 메시지를 보내면 이 함수가 실행됩니다.
@cl.on_message
async def on_message(recieved_message: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")

    send_message = cl.Message(content="")

    if chat_profile == "simple":
        async for chunk in LLM.astream(recieved_message.content):
            await send_message.stream_token(chunk.content)
        await send_message.send()

    elif chat_profile == "flight_assistant":
        # agent 실행시키기 위해 필요한 정보
        thread_id = str("12341234")
        config = {
            "configurable": {
                "passenger_id": "8498 685539",
                "thread_id": thread_id,
            }
        }

        # 화면에 보여지기 위한 설정으로 코드를 이해할 필요는 없습니다.
        step = set_step("agent_logs", recieved_message.content)

        # 이곳이 핵심!!!!!!!
        # Advanced Flight Assistant 실행되도록 하기
        memory = AsyncSqliteSaver.from_conn_string(":memory:")
        agent = builder.compile(checkpointer=memory)

        async for chunk in agent.astream(
            {"messages": ("user", recieved_message.content)},
            config,
            stream_mode="values",
        ):
            msg = chunk["messages"][-1].content
            name = chunk["messages"][-1].name or chunk["messages"][-1].type

            await step.stream_token("\n ====" + name + "==== \n")
            step.name = name
            step.update()

            await step.stream_token(msg)

        await step.send()
        await send_message.stream_token(msg)
        await send_message.send()

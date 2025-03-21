import os
import ssl
import httpx
import importlib
import chainlit as cl
import random
from literalai.helper import utc_now
from chainlit.step import Step
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


@cl.set_chat_profiles
async def chat_profile():
    files = os.listdir("./flows")
    flows = [f[:-3] for f in files if f.endswith(".py") and f != "__init__.py"]

    profiles = [
        cl.ChatProfile(
            name=flow,
            markdown_description=flow,
            icon="https://picsum.photos/{}".format(random.choice(range(200, 250))),
        )
        for flow in flows
    ]
    profiles.insert(
        0,
        cl.ChatProfile(
            name="default",
            markdown_description="일반",
            icon="https://picsum.photos/{}".format(random.choice(range(200, 250))),
        ),
    )
    return profiles


load_dotenv(override=True)

# Need Cleint with SSL Ceritificate
REQUEST_NEED_CERT_FILE = (
    os.environ.get("REQUEST_NEED_CERT_FILE", "False").lower() == "true"
)

client = None
if REQUEST_NEED_CERT_FILE:
    cert_file = os.environ.get("SSL_CERT_FILE", "./ssl_cacert.pem")
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=cert_file)
    client = httpx.Client(verify=ssl_context)

LLM_CLIENT = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")


MOVIE_REPO_IDS = [1]
KB_REPO_IDS = [2]


@cl.on_chat_start
async def start() -> None:
    chat_profile = cl.user_session.get("chat_profile")

    await cl.Message(
        content=f"`{chat_profile}` 설정에 기반하여 챗을 동작합니다.",
    ).send()


@cl.on_message
async def on_message(recieved_message: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "default":
        send_message = cl.Message(content="")
        async for chunk in LLM_CLIENT.astream(recieved_message.content):
            await send_message.stream_token(chunk.content)
        await send_message.send()

    else:
        flow = importlib.import_module(f"flows.{chat_profile}")
        app = flow.app
        thread = {"configurable": {"thread_id": "4"}}
        question = {"messages": [HumanMessage(content=recieved_message.content)]}
        send_message = cl.Message(content="")

        step = cl.Step(name="agent_logs")

        step.output = ""
        step = Step(
            name="agent_logs",
            type="tool",
        )
        step.start = utc_now()
        step.input = recieved_message.content

        async for event in app.astream(question, thread, stream_mode="values"):
            msg = event["messages"][-1].content
            name = event["messages"][-1].name or event["messages"][-1].type

            await step.stream_token("\n ====" + name + "==== \n")
            step.name = name
            step.update()

            await step.stream_token(msg)

        await step.send()
        await send_message.stream_token(msg)
        await send_message.send()

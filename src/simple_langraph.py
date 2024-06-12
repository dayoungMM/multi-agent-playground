import ssl

import httpx
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
from typing import Sequence

from langchain_core.messages import (
    BaseMessage,
)

from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
# from langgraph.prebuilt.tool_node import ToolNode


import os

load_dotenv()

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

# Define the tools for the agent to use
tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)

# Define the model

# client = AzureOpenAI(
#     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
#     api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
#     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
#     http_client=client,
#     timeout=60,
# )


model = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    http_client=client,
    timeout=60,
).bind_tools(tools)


# model = ChatOpenAI(temperature=0).bind_tools(tools)
# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep


# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # Use the Runnable
    # final_state = app.invoke(
    #     {"messages": [HumanMessage(content="what is the weather in Seoul Korea")]},
    #     config={"configurable": {"thread_id": 42}},
    # )
    # print(final_state)

    # run the graph
    thread = {"configurable": {"thread_id": "4"}}
    question = {
        "messages": [HumanMessage(content="what is the weather in Seoul Korea")]
    }

    for event in app.stream(question, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

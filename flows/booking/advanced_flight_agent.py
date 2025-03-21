import os
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Annotated

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from flows.booking.flight_tools import (
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from flows.booking.utils import create_tool_node_with_fallback, print_event


LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # configuration = config.get("configurable", {})
            # passenger_id = configuration.get("passenger_id", None)
            passenger_id = "3442 587242"
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# primary_assistant_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful customer support assistant for Swiss Airlines. "
#             " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             " If a search comes up empty, expand your search before giving up."
#             "passenger id: 3442 587242"
#             "\nCurrent time: {time}.",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now(), user_info="passenger id: 3442 587242")
prompt = """
You are a helpful customer support assistant for Swiss Airlines. 
Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 

If a search comes up empty, expand your search before giving up.
answer must be in Korean

passenger id: "8498 685539"
Current time: 2024.08.26
Customer Asked: {messages}
"""
primary_assistant_prompt = ChatPromptTemplate.from_template(prompt)

full_tools = [
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
]

part_1_assistant_runnable = primary_assistant_prompt | LLM.bind_tools(full_tools)


### Define Nodes
builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(full_tools))
# Define edges: these determine how the control flow moves
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = SqliteSaver.from_conn_string(":memory:")
advanced_agent_graph = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    tutorial_questions = [
        "Hi there, what time is my flight?",
        "Am i allowed to update my flight to something sooner? I want to leave later today.",
        "Update my flight to sometime next week then",
        "The next available option is great",
    ]
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "passenger_id": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    _printed = set()
    for question in tutorial_questions:
        events = advanced_agent_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            print_event(event, _printed)

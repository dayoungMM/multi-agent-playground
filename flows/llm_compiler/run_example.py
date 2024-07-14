import os
import ssl
import httpx

# from math_tools import get_math_tool
from langchain import hub
from langchain.chains import LLMMathChain
from langchain.tools import tool
from dotenv import load_dotenv
from flows.llm_compiler.planner import create_planner, plan_and_schedule
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.chains.openai_functions import create_structured_output_runnable
from flows.llm_compiler.joiner import (
    JoinOutputs,
    select_recent_messages,
    _parse_joiner_output,
)
from langgraph.graph import END, MessageGraph
from langchain_openai import ChatOpenAI

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

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

# llm = AzureChatOpenAI(
#     openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
#     azure_deployment=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME"),
#     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
#     http_client=client,
#     timeout=60,
# )


llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)


@tool
def calculator(query: str) -> str:
    """Make Numexpr from text. Calculate and return Answer"""
    input_data = {"question": query}
    calculate = LLMMathChain.from_llm(llm)
    return calculate.invoke(input_data)["answer"]


search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


@tool
def wikipedia_query_run(query: str) -> str:
    """Search Wikipedia for the query and return the first paragraph of the search result."""
    return search.invoke({"query": query})


def prep():
    tools = [wikipedia_query_run, calculator]

    tool_search_test = wikipedia_query_run.invoke(
        {"query": "What is GDP of South Korea?"}
    )
    tool_cal_test = calculator.invoke(
        {
            "query": "What is 3*4?",
        }
    )

    prompt = hub.pull("wfh/llm-compiler")
    print(prompt.pretty_print())
    planner = create_planner(llm, tools, prompt)

    example_question = "What is GDP of South Korea?"

    print(">>> Planner Example")
    for task in planner.stream([HumanMessage(content=example_question)]):
        print(task["tool"], task["args"])
        print("---")

    tool_messages = plan_and_schedule.invoke(
        [HumanMessage(content=example_question)], config={"planner": planner}
    )
    print(">>> Tool Messages")
    print(tool_messages)

    joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
        examples=""
    )  # You can optionally add examples

    runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)
    joiner = select_recent_messages | runnable | _parse_joiner_output

    input_messages = [HumanMessage(content=example_question)] + tool_messages
    join_result = joiner.invoke(input_messages)
    print(join_result)
    print("---" * 10)
    return planner, joiner


if __name__ == "__main__":
    planner, joiner = prep()

    graph_builder = MessageGraph()

    # 1.  Define vertices
    # We defined plan_and_schedule above already
    # Assign each node to a state variable to update
    graph_builder.add_node("plan_and_schedule", plan_and_schedule)
    graph_builder.add_node("join", joiner)

    ## Define edges
    graph_builder.add_edge("plan_and_schedule", "join")

    ### This condition determines looping logic

    def should_continue(state: list[BaseMessage]):
        if isinstance(state[-1], AIMessage):
            return END
        return "plan_and_schedule"

    graph_builder.add_conditional_edges(
        "join",
        should_continue,
    )
    graph_builder.set_entry_point("plan_and_schedule")
    chain = graph_builder.compile()

    print(">>> chain")

    for step in chain.stream(
        [
            HumanMessage(
                content="How big is GDP of South Korea compare to China, Taiwan, Japan"
            )
        ],
        config={"planner": planner},
    ):
        print(step)
        print("---")

    print(">>> final answer")
    print(step["join"][-1].content)

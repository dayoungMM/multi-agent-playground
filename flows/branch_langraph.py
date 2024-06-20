import os
import ssl
import httpx
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState

from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep

# from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
from typing import Sequence

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# from langgraph.prebuilt.tool_node import ToolNode


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


LLM_CLIENT = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    http_client=client,
    timeout=60,
)


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
def select_next_agent(
    state: AgentState,
) -> Literal["movie_agent", "interest_agent", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    # if last_message.tool_calls:
    #     return "tools"
    if "movie_agent" in last_message.content.lower():
        return "movie_agent"
    elif "interest_agent" in last_message.content.lower():
        return "interest_agent"
    else:
        auto_reply = AIMessage(
            content="주어진 질문에 대해 답변을 할 수 없습니다.", name="auto_reply"
        )
        state["messages"].append(auto_reply)
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = LLM_CLIENT.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def get_intent_agent_chain(state: AgentState) -> dict:
    messages = state["messages"]

    intent_agent_prompt_template = """Given the user question below, classify it 
    as either being about {classification_topics}
    Additional information to help decision:
    {descriptions}
    Do not respond with more than one word.
    <question>
    {query}
    </question>"""
    intent_agent_prompt = PromptTemplate.from_template(intent_agent_prompt_template)
    worker_agents = [
        ("movie_agent", "영화에 대한 질문을 대답할 수 있음"),
        ("kb_agent", "계약, 퇴직연금, 금융에 대해 대답할 수 있음"),
        ("general", "etc"),
    ]
    classification_topics_list = []
    descriptions = ""
    for name, desc in worker_agents:
        classification_topics_list.append(f"'{name}'")
        descriptions += f"{name} : {desc} \n"
    classification_topics = ",".join(classification_topics_list)

    parser = StrOutputParser()
    intent_sub_chain = (
        RunnablePassthrough.assign(
            classification_topics=lambda x: classification_topics,
            descriptions=lambda x: descriptions,
        )
        | intent_agent_prompt
        | LLM_CLIENT
        | parser
    )

    result = intent_sub_chain.invoke({"query": messages[-1]})
    return {"messages": [AIMessage(content=result, name="intent_agent")]}


def get_movie_agent_chain(state: AgentState) -> dict:
    from langchain.schema.output_parser import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    def retrieve(repo_ids: list[int], query: str):
        import json

        import requests

        print(">>>>> movie_searcher")

        try:
            HOST = os.environ.get("CORUS_HOST_URL")
            url = f"{HOST}/auth/v1/auth_device"
            api_key = os.environ.get("CORUS_DEVICE_KEY")

            payload = ""
            headers = {"User-Key": api_key, "Client-Type": "moviemate"}

            response = requests.request("POST", url, headers=headers, data=payload)
            access_token = response.json()["data"]["access_token"]
            print("ACCESS_TOKEN" + access_token)
            url = f"{HOST}/project/v1/repo/retrieval"

            payload = json.dumps(
                {
                    "ids": repo_ids,
                    "payload": {
                        "k": 3,
                        "query": query,
                        "threshold": 0.8,
                    },
                },
            )
            print("payload" + payload)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            }

            response = requests.request(
                "POST", url, headers=headers, data=payload, verify=False
            )
            if response.status_code == 200:
                repo_result = response.json()
                print(repo_result)
                return repo_result["results"]
            else:
                print("FAIL TO RETRIEVE")
                print(response)
                return "FAIL TO RETRIEVE" + str(response) + access_token
        except Exception as e:
            print(e)
            return "FAIL TO RETRIEVE" + str(e)

    REPO_IDS = [1]
    prompt_template = """
    #AI의 역할
당신은 내부 문서를 참고해 영화 관련 질문에 답하는 AI 비서입니다.
출처를 인용해 답변을 생성하는 경우 반드시 [REF-n]와 출처를 붙여서 인용을 표기하십시오.
답변은 한국말로 표시
질문: {query} 
출처: {agent_scratchpad}
답변:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # movie_sub_chain = RunnablePassthrough()
    movie_sub_chain = RunnablePassthrough.assign(
        agent_scratchpad=RunnableLambda(lambda x: retrieve(REPO_IDS, x["query"]))
    ) | RunnablePassthrough.assign(content=(prompt | LLM_CLIENT | StrOutputParser()))
    user_query = state["messages"][0].content

    result = movie_sub_chain.invoke({"query": user_query})

    return {"messages": [AIMessage(content=str(result), name="movie_agent")]}


def prep_interest_agent():
    import sqlite3

    import csv_to_sqlite

    csv_file_path = "./example_data/kb_interest.csv"
    db_file_path = "./example_data/kb.db"

    if not os.path.exists(csv_file_path):
        error_message = f"{csv_file_path} does not exist."
        raise FileNotFoundError(error_message)

    if not os.path.exists(db_file_path):
        error_message = "Database file does not exist."
        options = csv_to_sqlite.CsvOptions(typing_style="full", encoding="utf-8")
        input_files = [csv_file_path]
        csv_to_sqlite.write_csv(input_files, db_file_path, options)
        conn = sqlite3.connect(db_file_path)
        c = conn.cursor()
    assert True


def get_interest_agent_chain(state: AgentState) -> dict:
    prep_interest_agent()
    TABLE_INFO = """
        create table kb_interest
        (
            name     text, -- name of retirement pension product
            maturity text, -- maturity: Enum('1년','2년','3년')
            DB       text, -- 확정급여형(DB)
            DC       text, -- 확정기여형(DC)
            IRP      text, -- 개인형퇴직연금(IRP)
            supplier text -- provider of retirement pension product
        );
    """

    QUERY_TEMPLATE = """Make Query from user Question. 
        Create a sqlite query based on the TABLE_INFO
        Select the table name and column name included in the query to be included in the TABLE_INFO.
        Answer is in one sentence.

        TABLE_INFO: 
        {table_info}


        For example, 
        question: KB손보에서 나온 IRP중 1년만기 상품의 금리 찾아줘. 
        answer:
        ```sql
        select name, IRP from kb_interest where supplier ='KB손보' and maturity = '1년';
        ```
        question: KB손보의 퇴직연금 상품을 찾아줘.
        answer:
        ```sql
        select * from kb_interest where where supplier ='KB손보';
        ```

        Question: {query}
        Answer:
    """

    db = SQLDatabase.from_uri("sqlite:///src/example_data/kb.db")
    db_prompt = PromptTemplate.from_template(QUERY_TEMPLATE)

    db_chain = SQLDatabaseChain.from_llm(
        llm=LLM_CLIENT,
        db=db,
        verbose=True,
        use_query_checker=True,
        return_intermediate_steps=True,
    )
    final_chain = db_prompt | db_chain
    query = state["messages"][-1]["query"]
    result = final_chain.invoke(
        {
            "query": query,
            "table_info": TABLE_INFO,
        }
    )["result"]
    return {"messages": [AIMessage(content=result, name="interest_agent")]}


def get_solver(state: AgentState):
    messages = state["messages"]
    response = LLM_CLIENT.invoke(messages)
    return {"messages": [response]}
    # messages = state["messages"].append({"messages": [response]})
    # return END


# Define a new graph
workflow = StateGraph(MessagesState)


workflow.add_node("intent", get_intent_agent_chain)
workflow.add_node("interest_agent", get_interest_agent_chain)
workflow.add_node("movie_agent", get_movie_agent_chain)
workflow.add_node("solver", get_solver)


workflow.set_entry_point("intent")


workflow.add_conditional_edges(
    "intent",
    select_next_agent,
)
workflow.add_edge("movie_agent", "solver")
workflow.add_edge("interest_agent", "solver")
workflow.set_finish_point("solver")

checkpointer = MemorySaver()


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
    question = {"messages": [HumanMessage(content="이병헌이 나오는 영화 추천해줘")]}

    for event in app.stream(question, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

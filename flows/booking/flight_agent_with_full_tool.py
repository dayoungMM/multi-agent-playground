import os

from langchain_openai import ChatOpenAI
from flows.booking.flight_tools import (
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
)
from langchain.agents import AgentExecutor, create_structured_chat_agent

LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)

chat_prompt_template = """
Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation


 You are a helpful customer support assistant for Swiss Airlines. 
 Use the provided tools to search for flights, company policies, and other information to assist the user's queries.
 When searching, be persistent. Expand your query bounds if the first search returns no results.
 If a search comes up empty, expand your search before giving up.
 Current user: <User>{user_info}</User>
 Current time: {time}.

 {input}

{agent_scratchpad}
 (reminder to respond in a JSON blob no matter what. Answer must be in Korean)

"""

if __name__ == "__main__":
    # 이제 4개의 Tool을 모두 사용할 수 있는 Agent를 만들어보자

    full_tools = [
        fetch_user_flight_information,
        search_flights,
        update_ticket_to_new_flight,
        cancel_ticket,
    ]

    # TODO: ReACT Agent 만들어보기
    full_agent = None
    full_agent_executor = None
    full_agent = create_structured_chat_agent(LLM, full_tools, chat_prompt_template)
    full_agent_executor = AgentExecutor(agent=full_agent, tools=full_tools)
    result = full_agent_executor.invoke(
        {"input": "내 예약내역 알려줘"}, handle_parsing_errors=True
    )
    print(result)

    # 이 문제를 해결하기 위해서는 추론이 필요
    # Step 1: 다음주 ICN 에서 SHA로 가는 비행기 검색
    # step 2: ticke_no 7240005432906569 예약을 step1 의 flight id로 변경
    # 두 단계로 나눠서 해결해야 하는데, 단순 agent_executor로는 해결할 수 없음 - ReACT 한계
    # 에러 발생!!
    # 주석처리하세요

    result = full_agent_executor.invoke(
        {
            "input": "ticke_no 7240005432906569 예약을 다음주 ICN 에서 SHA로 가는 비행기로 변경해줘"
        },
        handle_parsing_errors=True,
    )
    print(result)

    # 대신 다음과 같이 추론이 필요하지 않은 질문은 잘 처리함
    result = full_agent_executor.invoke(
        {"input": "ticke_no 7240005432906569 예약을 flight id 23215로 변경해줘"},
        handle_parsing_errors=True,
    )
    print(result)
    result = full_agent_executor.invoke(
        {"input": "내 예약내역 알려줘"}, handle_parsing_errors=True
    )
    print(result)

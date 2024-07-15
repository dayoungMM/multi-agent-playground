import os
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from flows.booking.flight_tools import (
    fetch_user_flight_information,
    search_flights,
)

LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)

REACT_PROMPT_TEMPLATE = """
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


 You are a helpful customer support assistant for Swiss Airlines and Rental Car. 
 Use the provided tools to search for flights, company policies, and other information to assist the user's queries.
 When searching, be persistent. Expand your query bounds if the first search returns no results.
 If a search comes up empty, expand your search before giving up.
 Current user: <User>{user_info}</User>
 Current time: {time}.

 {input}

{agent_scratchpad}
 (reminder to respond in a JSON blob no matter what. Answer must be in Korean)

"""
# step 5: Let's put it all together
prompt = ChatPromptTemplate.from_template(REACT_PROMPT_TEMPLATE).partial(
    time=datetime.now(), user_info="passenger id: 3442 587242"
)
tools = [fetch_user_flight_information, search_flights]
agent = create_structured_chat_agent(LLM, tools, prompt)  # react agent를 만들어줌
agent_executor = AgentExecutor(
    agent=agent, tools=tools
)  # agent iteration 관리, 모니터링(callback handler), 에러 핸들링(ex: handle_parsing_errors)등 agent 실행에 필요한 기능 제공

if __name__ == "__main__":
    print(">>> Flight Assistant")
    result = agent_executor.invoke(
        {"input": "다음주에 ICN에서 SHA로 가는 비행일정 알려줘. flight_id 포함해서"},
        handle_parsing_errors=True,  # the error will be sent back to the LLM as an observation
    )
    print(result)
    print("---" * 10)

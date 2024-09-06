import os
from langserve import RemoteRunnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

joke_chain = RemoteRunnable("http://localhost:8080/joke/")

parrots_joke = joke_chain.invoke({"topic": "parrots"})
print(">>> Parrots Joke")
print(parrots_joke)
print("---")

agent = RemoteRunnable("http://localhost:8080/flight")

agent.invoke("온라인으로 취소 가능한 티켓은?")

retriever = RemoteRunnable("http://localhost:8080/retriever")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer Must be in Korean.
Always say "감사합니다." at the end of the answer.

Context: {context}

Question: {question}

Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)
llm_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | custom_rag_prompt | llm
)
llm_answer = llm_chain.invoke(
    "Which tickets/bookings cannot be rebooked online currently?"
)

#!/usr/bin/env python

import os
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from flows.booking.flight_policy import (
    EMBEDDING,
    format_docs,
    CHROMA_PATH,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

joke_prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)
joke_chain = joke_prompt | LLM

add_routes(
    app,
    joke_chain,
    path="/joke",
)


db_path = CHROMA_PATH
if not os.path.isdir(db_path):
    raise FileNotFoundError(f"The file at path {db_path} does not exist.")

db = Chroma(persist_directory=db_path, embedding_function=EMBEDDING)
retriever = db.as_retriever(
    search_type="mmr"
)  # as_retriever를 사용하면 VectorStoreRetriever로 리턴되는데, 이는 Runnable을 상속한 클래스
template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer Must be in Korean.
    Always say "감사합니다." at the end of the answer.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | LLM
    | StrOutputParser()
)

add_routes(
    app,
    rag_chain,
    path="/flight",
)

add_routes(app, retriever | format_docs, path="/retriever")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)

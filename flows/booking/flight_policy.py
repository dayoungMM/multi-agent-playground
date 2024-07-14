import os

# from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "./data/chroma_db"
EMBEDDING = OpenAIEmbeddings(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-3-large",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
    timeout=60,
)
LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)


def format_docs(docs):
    result = "\n\n".join(doc.page_content for doc in docs)
    return result


if __name__ == "__main__":
    # step 1: retriever 만들기
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING)
    retriever = db.as_retriever(search_type="mmr")

    retrieval_result = retriever.invoke(
        "Which tickets/bookings cannot be rebooked online currently?"
    )

    print(">>> Retrieval Result")
    print(retrieval_result)
    print("---" * 10)
    # step 2: retriever 결과 Formatting (Documents -> string)
    parsed_retrieval = retriever | format_docs
    context = parsed_retrieval.invoke(
        "Which tickets/bookings cannot be rebooked online currently?"
    )
    print(">>> Context")
    print(context)
    print("---" * 10)
    # step 3: llm에 질문할 Prompt 만들기.
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer Must be in Korean.
    Always say "감사합니다." at the end of the answer.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # context에 retriever 결과를, question는 유저 질문
    completed_prompt = custom_rag_prompt.invoke(
        {
            "context": context,
            "question": "Which tickets/bookings cannot be rebooked online currently?",
        }
    )
    print(">>> Completed Prompt")
    print(completed_prompt)
    print("---" * 10)

    # step 4: llm에 질문하기
    llm_answer = LLM.invoke(completed_prompt)
    print(">>> LLM Answer")
    print(llm_answer)
    print("---" * 10)

    # step 3 + step 4 + formatting
    llm_chain = custom_rag_prompt | LLM | StrOutputParser()
    llm_answer = llm_chain.invoke(
        {
            "context": context,
            "question": "Which tickets/bookings cannot be rebooked online currently?",
        }
    )
    print(">>> Formatted LLM Answer")
    print(llm_answer)
    print(">>>" * 10)

    ####################################
    # Now, let's put everything
    ####################################

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING)
    retriever = db.as_retriever(
        search_type="mmr"
    )  # as_retriever를 사용하면 VectorStoreRetriever로 리턴되는데, 이는 Runnable을 상속한 클래스

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | LLM
        | StrOutputParser()
    )

    final_result = rag_chain.invoke("현재 온라인으로 예약변경할 수 없는 경우는?")
    print(">>> Final Chain Result")
    print(final_result)
    print("---" * 10)

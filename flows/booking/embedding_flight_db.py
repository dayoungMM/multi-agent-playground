import os

# from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.chains import RetrievalQA
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "./data/chroma_db"

EMBEDDING = OpenAIEmbeddings(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-3-large",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
    timeout=60,
)
""" 
>>> how to test embedding
query_result = embeddings.embed_query("hello world")
"""

LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)


def create_chunks():
    loader = TextLoader("./data/swiss_faq.md")
    docs = loader.load()
    doc = docs[0].page_content
    doc = doc.replace(
        "\n\n", "\n\n<br>"
    )  # MarkdownHeaderTextSplitter에서 소실되지 않기 위해

    headers_to_split_on = [  # 문서를 분할할 헤더 레벨과 해당 레벨의 이름을 정의합니다.
        (
            "#",
            "Header 1",
        ),
        (
            "##",
            "Header 2",
        ),
        (
            "###",
            "Header 3",
        ),
    ]

    # 마크다운 헤더를 기준으로 텍스트를 분할하는 MarkdownHeaderTextSplitter 객체를 생성합니다.
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    # markdown_document를 헤더를 기준으로 분할하여 md_header_splits에 저장합니다.
    md_header_splits = markdown_splitter.split_text(doc)

    """ test if markdown_splitter works 
    for header in md_header_splits: 
        print(f"{header.page_content}")
        print(f"{header.metadata}", end="\n=====================\n")
    """

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\d+\.", "<br>"],  # 텍스트를 분할할 때 사용할 구분자를 지정합니다.
        keep_separator=False,  # 구분자를 결과에 포함할지 여부를 지정합니다.
        chunk_size=250,  # 분할된 텍스트 청크의 최대 크기를 지정합니다.
        chunk_overlap=0,  # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정합니다.
        length_function=len,  # 텍스트의 길이를 계산하는 함수를 지정합니다.
        is_separator_regex=True,  # 구분자가 정규식인지 여부를 지정합니다.
    )
    chunks = text_splitter.split_documents(md_header_splits)

    for c in chunks:
        print(f"{c.page_content}")
        print(f"{c.metadata}", end="\n=====================\n")
    return chunks


def create_index():
    # save to disk
    if not os.path.exists(CHROMA_PATH):
        chunks = create_chunks()
        db2 = Chroma.from_documents(chunks, EMBEDDING, persist_directory=CHROMA_PATH)


if __name__ == "__main__":
    create_index()

    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING)
    query = "Which tickets/bookings cannot be rebooked online currently?"
    docs = vectorstore.similarity_search(query, k=2)
    contents = [doc.page_content for doc in docs]

    print(">>> Query Result")
    print(contents)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

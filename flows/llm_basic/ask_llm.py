import os

# from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


if __name__ == "__main__":
    # llm = AzureChatOpenAI(
    #     openai_api_version=os.environ.get(
    #         "AZURE_OPENAI_API_VERSION", "2023-12-01-preview"
    #     ),
    #     azure_deployment=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME"),
    #     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    #     timeout=60,
    # )
    llm = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        organization=os.environ.get("OPENAI_ORGANIZATION"),
    )

    prompt = PromptTemplate.from_template("1 + {number} = ")
    chain = prompt | llm

    result = llm.invoke("hello")

    print(result.content)
    print("end")

    r = chain.invoke({"number": 3})
    print(r)

import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import httpx
import ssl

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


if __name__ == "__main__":
    llm = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        organization=os.environ.get("OPENAI_ORGANIZATION"),
        http_client=client,
    )

    prompt = PromptTemplate.from_template("1 + {number} = ")
    chain = prompt | llm

    result = llm.invoke("hello")

    print(result.content)
    print("end")

    r = chain.invoke({"number": 3})
    print(r)

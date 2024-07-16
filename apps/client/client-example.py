from langserve import RemoteRunnable


joke_chain = RemoteRunnable("http://localhost:8080/joke/")

parrots_joke = joke_chain.invoke({"topic": "parrots"})
print(">>> Parrots Joke")
print(parrots_joke)
print("---")

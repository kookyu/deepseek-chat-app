import ollama
client=ollama.Client(host="http://localhost:11434")
print(client.list())
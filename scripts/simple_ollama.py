import ollama  # type: ignore

# Simple call
response = ollama.chat(
    model="codellama:34b",
    messages=[{"role": "user", "content": 
        "Please implement the sieve of Eratosthenes in Python."}]
)
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(
        model="codellama:34b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True):
    print(chunk["message"]["content"], end="", flush=True)

# Embeddings
result = ollama.embeddings(model="nomic-embed-text", prompt="some text")
vec = result["embedding"]
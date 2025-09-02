from gpt4all import GPT4All

# This loads the specified model (will download if not present)
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

with model.chat_session():
    print(model.generate("give generic type of this item as one word or tag: trek super sport", max_tokens=50))
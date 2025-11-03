from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def add_chat(query, response):
    memory.save_context({"user": query}, {"llm": response})

def refresh():
    memory.clear()

def get_history():
    return memory.load_memory_variables({})['chat_history']
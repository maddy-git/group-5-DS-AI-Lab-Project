from langchain_classic.memory import ConversationBufferMemory

class Memory():
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.customer_id = -1

    def add_chat(self, query, response):
        self.memory.save_context({"user": query}, {"llm": response})

    def refresh(self):
        self.memory.clear()
        self.customer_id = -1

    def get_history(self):
        return self.memory.load_memory_variables({})['chat_history']

    def set_customer(self, id):
        self.customer_id = id

    def get_customer(self):
        return self.customer_id
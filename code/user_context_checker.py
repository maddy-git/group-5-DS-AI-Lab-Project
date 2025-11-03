from langchain_core.prompts import ChatPromptTemplate
from base_llm import llm
import json

extract_prompt = ChatPromptTemplate.from_template("""
You are an assistant that extracts Age and Gender from a user's message.
If not mentioned, return empty values.

User message: {user_input}

Respond ONLY in JSON format like:
{{"Age": "23", "Gender": "Male"}}
""")

extract_chain = extract_prompt | llm


# checks whether there is sufficient info available like age and gender
from langchain_core.prompts import MessagesPlaceholder

check_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a fashion assistant.
Check if the user's context and chat history include both Age and Gender.
Rules:
- If Age or Gender is missing, politely ask the user to provide them.
- If both are present, respond with: 'All required info is available.'"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{context}")
])

check_chain = check_prompt | llm


# --- Step 3: Simple conversational flow ---
def handle_user_query(user_query, user_context, memory):
    """
    Extracts new info (age/gender), updates context,
    checks sufficiency, and replies accordingly.
    """

    # gives the user query to extract age and gender and return a json file
    extraction = extract_chain.invoke({"user_input": user_query})
    print(f"🧾 Extraction:\n{extraction.content}\n")

    try:
        #updates the context if the user provided a age/gender in his promt
        start_index = extraction.content.find('{')
        end_index = extraction.content.find('}') + 1
        info = json.loads(extraction.content[start_index:end_index])
        if info.get("Age"):
            user_context = [c for c in user_context if not c.startswith("Age:")]
            user_context.insert(0, f"Age: {info['Age']}")
        if info.get("Gender"):
            user_context = [c for c in user_context if not c.startswith("Gender:")]
            user_context.insert(0, f"Gender: {info['Gender']}")
    except Exception:
        pass

    # Step 2: Combine context + chat history
    chat_history_text = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in memory.get_history()]
    )
    combined_info = "\n".join(user_context) + "\n\n" + chat_history_text

    # Step 3: Check sufficiency
    sufficiency_check = check_chain.invoke({
        "context": combined_info,
        "chat_history": memory.get_history()
    })

    print(f"🤖 Sufficiency check:\n{sufficiency_check.content}\n")
    return sufficiency_check.content

    # # Step 4: Decide reply
    # if "ask" in sufficiency_check.content.lower() or "provide" in sufficiency_check.content.lower():
    #     return sufficiency_check.content, user_context
    # else:
    #     # If sufficient info, just echo or respond normally
    #     if user_query.lower() in ["hi", "hello", "hey"]:
    #         return "Hello! 👋", user_context
    #     else:
    #         return f"You said: {user_query}", user_context


# # --- Step 4: Test flow ---
# context = [
#     "--- CUSTOMER CONTEXT ---",
#     "Age: ",      # missing initially
#     "Gender: ",   # missing initially
#     "---------------------------"
# ]
#
# memory = Memory()
#
# # 1️⃣ User says hi
# print("👤 User: hi\n")
# response, context = handle_user_query("hi", context, memory)
# print("💬 Bot:", response, "\n")
#
# # 2️⃣ User provides partial info
# print("👤 User: my gender is female\n")
# response, context = handle_user_query("my gender is female", context, memory)
# print("💬 Bot:", response, "\n")
#
# # 3️⃣ User provides remaining info
# print("👤 User: I am 24 years old\n")
# response, context = handle_user_query("I am 24 years old", context, memory)
# print("💬 Bot:", response, "\n")
#
# # 4️⃣ User says something again
# print("👤 User: hello\n")
# response, context = handle_user_query("hello", context, memory)
# print("💬 Bot:", response, "\n")
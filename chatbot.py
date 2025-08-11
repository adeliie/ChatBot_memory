from langchain_core.messages import SystemMessage,  AIMessage
from config import llm, mem0
from memory_db import query_db, load_user_context
from utils import format_conv

def chatbot(state):
    messages = state["messages"]
    user_id = state["mem0_user_id"]
    history = state["chat_history"]
    memories = mem0.search(messages[-1].content, user_id=user_id)
    context = "".join(f"- {memory['memory']}\n" for memory in memories)
    query_res = query_db(state, user_id, messages[-1].content)
    doc_db = query_res
    user_context = load_user_context(user_id)
    recent_history = history[-10:-1] if len(history) > 1 else []
    system_message = SystemMessage(content=f"""I\nYou are a helpful and knowledgeable assistant for Chatbotting. Your goal is to provide accurate, relevant, and contextually appropriate answers to user queries.\nIf the user's message is not a question then just answer as a joyfull and useful chatbot to converse.\n                                   \n##Context set by the user about what/how do they want you to answer:\n- **User Context**: {user_context}\n## Response Scope\n- **Answer only the user's message.** Do not add meta-commentary like \"your message is a greeting\" or explanations about how you're using information.\n- **Do not mention the existence of documents or prior context** in your response — just use them naturally if relevant.\n- If the user input is empty, vague, or unclear, DO NOT GUESS. \n\n## Information Sources\nOnly use these when they directly help answer a specific question:\n1. **CONTEXT** – Information from the user's past interactions. Likely information includes preferences, interests, and details about the user. Do not use them if the user's query is not related to these past interactions.\n2. **DOCUMENT** – Extracts from the user's personal document database. They likely do not contain personal information.\n3. **CONVERSATION** Your current conversation with the user.\n                                   \n## Response Guidelines\n-The response should be concise, relevant, and directly address the user's query. Don't add information about the user or past conversations just because you have them is it doesn't help to ake a better answer.\n\n### Use CONTEXT/DOCUMENT only when:\n- It adds clarity, precision, or relevance to the user's specific query\n- It helps answer questions involving prior discussions or uploaded content\n\n### Avoid using CONTEXT/DOCUMENT when:\n- Do not use the documents/context juse to add information if it's not direclty linked to the user's query\n- The user is making small talk or casual comments\n- The user's question is clear and complete without background info\n\n---\n\n### CONTEXT:\n{context}\n\n### DOCUMENT (ranked by relevance score - higher scores are more relevant):\n{doc_db}\n\n---\n### CURRENT CONVERSATION WITH THE USER: :\n{format_conv(recent_history)}\n\n\n### USER QUERY:{messages[-1].content}\n""")   
    full_messages = [system_message] 
    response = AIMessage(llm.invoke(full_messages))
    mem0.add([
        {"role": "user", "content": messages[-1].content},
        {"role": "assistant", "content": response.content},
    ], user_id=user_id, output_format="v1.1")
    return {"messages": [response]}

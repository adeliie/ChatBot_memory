from langchain_core.messages import HumanMessage, AIMessage
from graph_setup import compiled_graph
from typing import List
from memory_db import get_chroma

def run_conversation(user_input: str, mem0_user_id: str, chat_history: List = None):
    config = {"configurable": {"thread_id": mem0_user_id}}
    chat_history = chat_history or []
    chat_history.append(HumanMessage(content=user_input))

    state = {
        "messages": chat_history,
        "mem0_user_id": mem0_user_id,
        "chat_history": chat_history,
    }

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            response = value.get("messages")[-1]
            if isinstance(response, AIMessage):
                print("AI:", response.content)
                chat_history.append(response)
    
    return chat_history

if __name__ == "__main__":
    user_id = input("What is your ID? ").strip()
    CHROMA_PATH = get_chroma(user_id)
    
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("AI: Thank you for contacting us. Goodbye!")
            break
        chat_history = run_conversation(user_input, user_id, chat_history)

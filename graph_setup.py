from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated, TypedDict, List
from chatbot import chatbot
from document import load_document
from memory_db import save_user_context

def user(state):
    return state

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str
    chat_history: List[HumanMessage | AIMessage] = []

def conditional_router(state: State):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage) and "load document" in last_msg.content.lower():
        return "load_document"
    elif isinstance(last_msg, HumanMessage) and "set_context" in last_msg.content.lower():
        return "prompt_user"
    return "chatbot"

graph = StateGraph(State)
graph.add_node("user", user)
graph.add_node("chatbot", chatbot)
graph.add_node("load_document", load_document)
graph.add_node("prompt_user", save_user_context)
graph.add_edge(START, "user")
graph.add_conditional_edges("user", conditional_router)
compiled_graph = graph.compile()

from langchain_core.messages import HumanMessage, AIMessage
from typing import List

def format_conv(messages: List[HumanMessage | AIMessage]):
    return '\n'.join([f"{m.type}: {m.content}" for m in messages])


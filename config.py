from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from mem0 import MemoryClient
import torch
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# Load API key from environment variable
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")  

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cuda",  # or "cpu" as needed
        "trust_remote_code": True
    }
)
llm = OllamaLLM(model="llama3", temperature=0.3)
mem0 = MemoryClient(api_key=MEM0_API_KEY)

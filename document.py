from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import embedding_function, llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma
from memory_db import get_chroma


def load_document(state):
    CHROMA_PATH = get_chroma(state["mem0_user_id"])
    mess = state["messages"][-1].content
    file_path = mess.replace("load document", "").strip()
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    doc_type = guess_type(state, documents)
    print(f"Document type guessed: {doc_type}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = splitter.split_documents(documents)
    
    documents = [Document(page_content=doc.page_content, metadata={"type": doc_type}) for doc in documents]
    
    db = Chroma.from_documents(documents, embedding_function, persist_directory=CHROMA_PATH)
    response = AIMessage(content=f"📄 Document '{file_path}' chargé avec succès.")
    return {"messages": [response]}


def guess_type(state, documents):
    """Guess the type of the documents based on their content"""
    if not documents:
        return "No documents available."
    
    # Limit to first 3 documents and reduce content length for faster processing
    doc_content = ''.join([doc.page_content[:1000] for doc in documents[:3]])  # Limit content length
    
    messages = [
        SystemMessage(content="You are a helpful assistant that classifies documents based on their content "
        "and add a little description of the document. Just answer the type and description of the " \
        "following document nothing else. The description should be 1 sentence max"),
        HumanMessage(content=f"""DOCUMENT: {doc_content}"""),
    ]   
    
    response = llm.invoke(messages)
    response = response.strip()
    if not response:
        return "Unable to classify documents."

    return response


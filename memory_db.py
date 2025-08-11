from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from config import embedding_function
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
from pathlib import Path
from functools import lru_cache
from langchain_core.messages import  HumanMessage, AIMessage

def get_chroma(user_id):
    return f"chroma_db/{user_id}"

def query_db(state, user_id: str, query: str):
    try:
        # Use cached database connection
        user_context = embedding_function.embed_query(load_user_context(user_id))
       
        db = Chroma(persist_directory=f"chroma_db/{user_id}", embedding_function=embedding_function)
        
        docs = db.get()
        if not docs["documents"]:
            return "No documents available."

        all_docs = [Document(page_content=doc, metadata=meta or {}) for doc, meta in zip(docs["documents"], docs["metadatas"])]
        
        recent_messages = state["messages"][-6:]
        conv_profile = "\n".join(m.content for m in recent_messages if isinstance(m, HumanMessage) and "load_document" not in m.content and "set_context" not in m.content)
        
        conv_embedding = embedding_function.embed_query(conv_profile)
    
        meta_embeddings = [embedding_function.embed_query(doc.metadata.get("type", "")) for doc in all_docs if doc.metadata]
    
        similarities = cosine_similarity([conv_embedding], meta_embeddings)[0]
        context_similarities = cosine_similarity([user_context], [embedding_function.embed_query(doc.page_content) for doc in all_docs])[0]
        threshold = 0.3
        relevant_docs = [doc for doc, sim in zip(all_docs, similarities) if sim >= threshold]

        if not relevant_docs:
            return "No relevant documents found."

        bm25_retriever = BM25Retriever.from_documents(relevant_docs, k=min(10, len(relevant_docs)))
        bm25_results = bm25_retriever.invoke(query)

    
        chroma_retriever = db.as_retriever(search_kwargs={"k": min(10, len(relevant_docs))})
        chroma_results = chroma_retriever.invoke(query)

        relevant_docs_content = {doc.page_content for doc in relevant_docs}
        chroma_scored = {
            i: (doc, (len(chroma_results) - i) / len(chroma_results))
            for i, doc in enumerate(chroma_results)
            if doc.page_content in relevant_docs_content
        }

        all_doc_indices = set(range(len(bm25_results))) | set(chroma_scored.keys())
        final_scored = []
        
        for doc_idx in all_doc_indices:
            if doc_idx < len(bm25_results):
                bm25_doc = bm25_results[doc_idx]
                bm25_score = (len(bm25_results) - doc_idx) / len(bm25_results)
            else:
                bm25_doc = None
                bm25_score = 0.0
                
            if doc_idx in chroma_scored:
                chroma_doc, chroma_score = chroma_scored[doc_idx]
                if bm25_doc is None:
                    doc = chroma_doc
                else:
                    doc = bm25_doc
            else:
                chroma_score = 0.0
                doc = bm25_doc
            
            try:
                doc_index = next(i for i, d in enumerate(all_docs) if d.page_content == doc.page_content)
                semantic_score = similarities[doc_index]
                context_score = context_similarities[doc_index]
            except (StopIteration, IndexError):
                semantic_score = 0.0
            
            retriever_combined_score = (bm25_score * 0.4 + chroma_score * 0.6)
            final_score = (retriever_combined_score * 0.6 + semantic_score * 0.3 + context_score * 0.1)
            final_scored.append((doc, final_score))

        final_scored.sort(key=lambda x: x[1], reverse=True)
        results = [doc.page_content for doc, score in final_scored[:10]]

        if not results:
            return "No relevant documents found."

        return "\n".join(results)

    except Exception as e:
        print(f"Erreur dans query_db: {e}")
        return "Error querying database."

@lru_cache(maxsize=128)
def get_context_path(user_id):
    context_dir = Path("user_contexts")
    context_dir.mkdir(exist_ok=True)
    return context_dir / f"{user_id}_context.json"

def save_user_context(state):
    user_id = state["mem0_user_id"]
    context = state["messages"][-1].content.strip().replace("set_context", "").strip()
    path = get_context_path(user_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"context": context, "timestamp": time.time()}, f, indent=2, ensure_ascii=False)
    return {"messages": [AIMessage(content = "✅ Votre contexte utilisateur a été mis à jour.")]}

def load_user_context(user_id: str):
    path = get_context_path(user_id)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get("context", "")
    return ""

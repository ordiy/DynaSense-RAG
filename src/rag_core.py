import os
import json
import math
import asyncio
import requests
import lancedb
import mongomock
from typing import List, Dict, TypedDict, Tuple
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

# --- Configuration ---
JINA_API_KEY = os.environ.get("JINA_API_KEY", "jina_fc4562cfaf284cea94b01af1294d13c0LW8z0DPSfQJp8jkfVUeoAbVpIT40")
LANCEDB_URI = os.environ.get("LANCEDB_URI", "./data/lancedb_store")
TABLE_NAME = "knowledge_base"

# --- Initialization ---
os.makedirs(os.path.dirname(LANCEDB_URI), exist_ok=True)
doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
query_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
llm = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0)

# MongoDB Mock setup
mongo_client = mongomock.MongoClient()
db_mongo = mongo_client.doc_db
collection = db_mongo.chunks

# LanceDB setup
db_lance = lancedb.connect(LANCEDB_URI)
try:
    vectorstore = LanceDB(connection=db_lance, table_name=TABLE_NAME, embedding=doc_embeddings)
except:
    vectorstore = None # Will be created on first insertion


# --- Utilities ---
def chunk_text_jina(text: str) -> List[str]:
    url = "https://segment.jina.ai/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    formatted_text = text.replace("。 ", "。\n\n").replace("：", "：\n\n")
    data = {"content": formatted_text, "return_chunks": True}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        res_json = response.json()
        chunks = res_json.get("chunks", [text])
        return [c.strip() for c in chunks if c.strip()]
    except Exception as e:
        print(f"Jina segmentation failed: {e}")
        return [text]

def jina_rerank(query: str, retrieved_docs: List[Document], top_n: int = 2) -> List[Document]:
    if not retrieved_docs:
        return []
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    doc_texts = [doc.page_content for doc in retrieved_docs]
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": doc_texts,
        "top_n": top_n
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        res_json = response.json()
        reranked_docs = []
        for result in res_json["results"]:
            original_index = result["index"]
            reranked_docs.append(retrieved_docs[original_index])
        return reranked_docs
    except Exception as e:
        print(f"Jina Rerank failed: {e}")
        return retrieved_docs[:top_n] 

# --- Async Ingestion Task ---
async def process_document_task(content: str, filename: str, task_state: dict):
    global vectorstore
    task_state["status"] = "chunking"
    task_state["progress"] = 20
    
    # Simulate IO and perform Jina chunking
    await asyncio.sleep(0.1)
    chunks = chunk_text_jina(content)
    
    task_state["status"] = "embedding_and_indexing"
    task_state["progress"] = 50
    
    docs_to_insert = []
    mongo_records = []
    
    # Assign unique IDs
    start_id = collection.count_documents({}) + 1
    
    for i, chunk in enumerate(chunks):
        doc_id = start_id + i
        meta = {"doc_id": doc_id, "source": filename}
        docs_to_insert.append(Document(page_content=chunk, metadata=meta))
        mongo_records.append({"id": doc_id, "source": filename, "content": chunk})
        
    # Insert to Mongo
    if mongo_records:
        collection.insert_many(mongo_records)
    
    task_state["progress"] = 70
    
    # Insert to LanceDB
    if docs_to_insert:
        if TABLE_NAME not in db_lance.table_names():
            vectorstore = LanceDB.from_documents(docs_to_insert, doc_embeddings, connection=db_lance, table_name=TABLE_NAME)
        else:
            vectorstore.add_documents(docs_to_insert)

    task_state["status"] = "completed"
    task_state["progress"] = 100
    task_state["result"] = f"Processed {len(chunks)} chunks."

# --- Retrieval Pipeline (LangGraph) ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    loop_count: int
    logs: List[str]

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes or no")

grader_llm = llm.with_structured_output(GradeDocuments)

def retrieve_and_rerank_node(state: AgentState):
    state.get("logs", []).append("Executing Vector Search (Top 10)")
    if not vectorstore:
        return {"documents": [], "logs": state.get("logs", [])}
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    retrieved_docs = retriever.invoke(state["question"])
    
    state.get("logs", []).append("Executing Jina Cross-Encoder Rerank (Top 3)")
    reranked_docs = jina_rerank(state["question"], retrieved_docs, top_n=3)
    
    return {"documents": [doc.page_content for doc in reranked_docs], "logs": state.get("logs", [])}

def grade_documents_node(state: AgentState):
    state.get("logs", []).append("Grading context to prevent hallucination...")
    filtered_docs = []
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "判断给定的文档是否包含解答问题的关键信息。如果不相关，必须回答 'no'。"),
        ("human", "问题: {question}\n\n文档: {document}")
    ])
    for doc in state["documents"]:
        try:
            result = grader_llm.invoke(grade_prompt.format_messages(question=state["question"], document=doc))
            if result and result.binary_score.lower() == "yes":
                filtered_docs.append(doc)
        except:
            filtered_docs.append(doc)
            
    if not filtered_docs:
        state.get("logs", []).append("⚠️ All contexts rejected. Hallucination blocked.")
    else:
        state.get("logs", []).append(f"✅ {len(filtered_docs)} contexts approved.")
        
    return {"documents": filtered_docs, "logs": state.get("logs", [])}

def generate_node(state: AgentState):
    if not state["documents"]:
        return {"generation": "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。"}
        
    state.get("logs", []).append("Generating answer based on verified context...")
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "严格使用以下检索到的上下文来回答问题。绝不允许编造。\n上下文: {context}"),
        ("human", "问题: {question}")
    ])
    generation = llm.invoke(gen_prompt.format_messages(context="\n".join(state["documents"]), question=state["question"])).content
    return {"generation": generation, "logs": state.get("logs", [])}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_and_rerank_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)
rag_app = workflow.compile()

def run_chat_pipeline(query: str):
    inputs = {"question": query, "loop_count": 0, "logs": []}
    result = rag_app.invoke(inputs)
    return {
        "answer": result["generation"],
        "context_used": result["documents"],
        "logs": result["logs"]
    }

# --- Evaluation Logic ---
def calculate_ndcg(hit_rank: int, k: int) -> float:
    """Calculate NDCG@K. hit_rank is 0-indexed."""
    if hit_rank == -1 or hit_rank >= k:
        return 0.0
    return 1.0 / math.log2(hit_rank + 2) # rank+1 for 1-based, +1 for formula = rank+2

def run_evaluation(query: str, expected_substring: str):
    if not vectorstore:
        return {"error": "Database is empty."}
        
    # Vector Search
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    base_docs = retriever.invoke(query)
    
    # Rerank
    reranked_docs = jina_rerank(query, base_docs, top_n=10)
    
    # Find rank
    hit_rank = -1
    for i, doc in enumerate(reranked_docs):
        if expected_substring.lower() in doc.page_content.lower():
            hit_rank = i
            break
            
    results = {}
    for k in [1, 3, 5, 10]:
        recall = 1 if (hit_rank != -1 and hit_rank < k) else 0
        ndcg = calculate_ndcg(hit_rank, k)
        results[f"K={k}"] = {"Recall": recall, "NDCG": round(ndcg, 3)}
        
    return {
        "query": query,
        "expected_substring": expected_substring,
        "hit_rank": hit_rank + 1 if hit_rank != -1 else "Not Found",
        "metrics": results,
        "top_3_docs": [d.page_content for d in reranked_docs[:3]]
    }

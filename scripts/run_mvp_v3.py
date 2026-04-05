import os
import sys
import json
import requests
from pathlib import Path
from typing import List, TypedDict
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not os.environ.get("DATABASE_URL"):
    raise SystemExit("Set DATABASE_URL for PostgreSQL + pgvector.")

from src.infrastructure.persistence.postgres_connection import init_pool, get_pool
from src.infrastructure.persistence.postgres_schema import ensure_schema, truncate_kb_storage
from src.infrastructure.persistence.postgres_vectorstore import PostgresVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

# --- Config ---
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
if not JINA_API_KEY:
    raise SystemExit("Set JINA_API_KEY (do not commit secrets to git).")

print("Initializing models...")
doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
query_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
llm = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0)

# --- Document Chunking via Jina Segmenter ---
def chunk_text_jina(text: str) -> List[str]:
    url = "https://segment.jina.ai/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    formatted_text = text.replace("。 ", "。\n\n").replace("：", "：\n\n")
    data = {
        "content": formatted_text,
        "return_chunks": True
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        res_json = response.json()
        chunks = res_json.get("chunks", [text])
        return [c.strip() for c in chunks if c.strip()]
    except Exception as e:
        print(f"Jina segmentation failed: {e}")
        return [text]

# --- Reranking with Jina Multilingual Reranker ---
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

# --- Data Ingestion ---
print("Preparing dataset and chunking via Jina...")
raw_docs = [
    {"id": 1, "text": "公司 2023 年 Q3 营收为 500 万美元。主要得益于 AI 产品的订阅增长。"},
    {"id": 2, "text": "2024 年战略规划：全面投入 Agentic 架构研发，特别是 MAP-RAG 系统。"},
    {"id": 3, "text": "我们的旗舰产品 'DataSphere' 用户量在上个月突破了 10 万大关。这是历史性的时刻。"},
    {"id": 4, "text": "人力资源部通知：本周五下午将举行年度团建活动，地点在奥林匹克公园。请准时参加。"},
    {"id": 5, "text": "技术预研报告：关于新一代 Matryoshka 表征学习在向量搜索中的降本增效分析。该技术有望降低80%存储。"},
    {"id": 6, "text": "客户反馈汇总：近期有 15% 的用户抱怨新版本的加载速度变慢，需要前端团队重点关注性能优化。"},
    {"id": 7, "text": "财务审批流程更新：所有超过 1000 美元的采购都需要经过部门经理和 CFO 的双重审批。不可越权。"},
    {"id": 8, "text": "开源社区贡献指南：我们鼓励员工在工作时间参与开源项目，并将其作为绩效考核的一部分。"},
    {"id": 9, "text": "服务器迁移计划：预计在下个月中旬，我们将所有的后端服务从 AWS 迁移到 Google Cloud GCP。"},
    {"id": 10, "text": "新员工入职培训：每月的第一个星期一为新员工统一入职培训日，涵盖公司文化和基础安全操作规程。"}
]

chunked_documents = []
chunk_id_counter = 1

for doc in raw_docs:
    # Adding metadata optimization for reranker
    enriched_text = f"[类别: 公司内部文档] {doc['text']}" 
    chunks = chunk_text_jina(enriched_text)
    for chunk_text in chunks:
        chunked_documents.append(
            Document(
                page_content=chunk_text, 
                metadata={"id": chunk_id_counter, "parent_id": doc["id"]}
            )
        )
        chunk_id_counter += 1

print(f"Original docs: {len(raw_docs)}, Total chunks after Jina: {len(chunked_documents)}")

print("Indexing to PostgreSQL kb_embedding (vectors)...")
init_pool(os.environ["DATABASE_URL"])
ensure_schema(get_pool())
truncate_kb_storage(get_pool())
vs = PostgresVectorStore(get_pool(), doc_embeddings)
texts = [d.page_content for d in chunked_documents]
embeds = doc_embeddings.embed_documents(texts)
rows = [
    (str(d.metadata["id"]), d.page_content, dict(d.metadata), embeds[j])
    for j, d in enumerate(chunked_documents)
]
vs.add_embedding_rows(rows)
vectorstore = vs

# Fetch Top K=10 to give Cross-Encoder more candidate pool
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# --- Evaluation ---
def run_evaluation_v3():
    print("\n========== v1.2 开始评估 Recall 指标 (Chunking + Jina Semantic Rerank) ==========")
    test_queries = [
        {"query": "咱们公司未来打算搞什么技术架构？", "expected_parent_id": 2},
        {"query": "公司23年三季度的营收情况", "expected_parent_id": 1},
        {"query": "降维向量的相关的技术分析", "expected_parent_id": 5},
        {"query": "星期五有什么安排？", "expected_parent_id": 4},
        {"query": "前端新版本加载慢怎么回事？", "expected_parent_id": 6},
        {"query": "入职培训什么时间？", "expected_parent_id": 10}
    ]

    correct_retrievals = 0
    results_data = []
    
    for t in test_queries:
        # Get K=10 from vector
        initial_docs = vectorstore.similarity_search(t["query"], k=10)
        # Rerank to Top=2 using real Jina Reranker
        final_docs = jina_rerank(t["query"], initial_docs, top_n=2)
        
        retrieved_parent_ids = [d.metadata["parent_id"] for d in final_docs]
        is_hit = t["expected_parent_id"] in retrieved_parent_ids
        if is_hit:
            correct_retrievals += 1
            
        print(f"Query: '{t['query']}' -> Parent IDs: {retrieved_parent_ids}, Expected: {t['expected_parent_id']}, Hit: {is_hit}")
        
        results_data.append({
            "query": t["query"],
            "expected_parent_id": t["expected_parent_id"],
            "retrieved_parent_ids": retrieved_parent_ids,
            "hit": is_hit
        })

    recall_rate = correct_retrievals / len(test_queries)
    print(f"Recall@2 (Post-Rerank): {correct_retrievals}/{len(test_queries)} ({recall_rate*100:.2f}%)")
    
if __name__ == "__main__":
    run_evaluation_v3()

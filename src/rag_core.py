"""
RAG core: LangGraph pipeline, embeddings, LanceDB.

LangSmith: call ``src.observability.init_langsmith_tracing()`` before this module
is imported (``app.py`` does this for the API server). Standalone scripts should
call it first if tracing is desired.
"""
import os
import json
import time
import uuid
import requests
import lancedb
import mongomock
from typing import List, Dict, TypedDict, Tuple, Any
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

import logging

# Configure basic logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Configuration ---
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
LANCEDB_URI = os.environ.get("LANCEDB_URI", "./data/lancedb_store")
TABLE_NAME = "knowledge_base"

# Jina 外部调用：超时/重试，避免 worker 长时间阻塞
JINA_REQUEST_TIMEOUT = (5, 20)  # (connect_timeout, read_timeout)
JINA_MAX_RETRIES = 3

# --- Initialization ---
os.makedirs(os.path.dirname(LANCEDB_URI), exist_ok=True)
doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
query_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)

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

_http_session = requests.Session()
if not JINA_API_KEY:
    logger.warning("`JINA_API_KEY` is not set; chunking/reranking via Jina will be degraded (fallback behavior).")


# --- Utilities ---
def _post_json_with_retries(url: str, headers: dict, payload: dict) -> dict:
    last_err: Exception | None = None
    for attempt in range(JINA_MAX_RETRIES):
        try:
            resp = _http_session.post(url, headers=headers, json=payload, timeout=JINA_REQUEST_TIMEOUT)
            # 常见可重试状态码（含限流）
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(2**attempt)
    raise RuntimeError(f"Jina request failed after retries: {last_err}")

def chunk_text_jina(text: str) -> List[str]:
    if not JINA_API_KEY:
        return [text]

    url = "https://segment.jina.ai/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    formatted_text = text.replace("。 ", "。\n\n").replace("：", "：\n\n")
    data = {"content": formatted_text, "return_chunks": True}
    try:
        res_json = _post_json_with_retries(url, headers=headers, payload=data)
        chunks = res_json.get("chunks", [text])
        return [c.strip() for c in chunks if c.strip()]
    except Exception as e:
        logger.error(f"Jina segmentation failed: {e}")
        return [text]

def jina_rerank(query: str, retrieved_docs: List[Document], top_n: int = 2) -> List[Document]:
    if not retrieved_docs:
        return []
    if not JINA_API_KEY:
        return retrieved_docs[:top_n]
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
        res_json = _post_json_with_retries(url, headers=headers, payload=data)
        reranked_docs = []
        for result in res_json["results"]:
            original_index = result["index"]
            reranked_docs.append(retrieved_docs[original_index])
        return reranked_docs
    except Exception as e:
        logger.error(f"Jina Rerank failed: {e}")
        return retrieved_docs[:top_n] 

# --- Ingestion Task ---
def process_document_task(content: str, filename: str, task_state: dict):
    global vectorstore
    task_state["status"] = "chunking"
    task_state["progress"] = 20
    
    chunks = chunk_text_jina(content)
    
    task_state["status"] = "embedding_and_indexing"
    task_state["progress"] = 50
    
    docs_to_insert = []
    mongo_records = []
    
    # Implement Small-to-Big: Store parent doc
    parent_id = f"parent_{uuid.uuid4().hex[:8]}"
    collection.insert_one({
        "id": parent_id,
        "type": "parent",
        "source": filename,
        "full_content": content
    })
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{parent_id}_{i}"
        meta = {"doc_id": chunk_id, "parent_id": parent_id, "source": filename}
        docs_to_insert.append(Document(page_content=chunk, metadata=meta))
        mongo_records.append({
            "id": chunk_id, 
            "type": "child",
            "parent_id": parent_id, 
            "source": filename, 
            "content": chunk
        })
        
    # Insert to Mongo
    if mongo_records:
        collection.insert_many(mongo_records)
    
    task_state["progress"] = 70
    
    # Insert to LanceDB
    if docs_to_insert:
        # We manually embed to avoid Langchain LanceDB wrapper bugs when appending
        texts = [d.page_content for d in docs_to_insert]
        embeds = doc_embeddings.embed_documents(texts)
        data = []
        for j, d in enumerate(docs_to_insert):
            data.append({
                "vector": embeds[j], 
                "text": d.page_content, 
                "metadata": d.metadata
            })
            
        if TABLE_NAME in db_lance.table_names():
            tbl = db_lance.open_table(TABLE_NAME)
            tbl.add(data)
            if vectorstore is None:
                vectorstore = LanceDB(connection=db_lance, table_name=TABLE_NAME, embedding=doc_embeddings)
        else:
            tbl = db_lance.create_table(TABLE_NAME, data=data)
            vectorstore = LanceDB(connection=db_lance, table_name=TABLE_NAME, embedding=doc_embeddings)

    # Neo4j: LLM triple extraction + merge (MVP; optional if DB down / benchmark skip)
    if os.environ.get("SKIP_NEO4J_INGEST", "").lower() in ("1", "true", "yes"):
        pass
    else:
        try:
            from src.hybrid_rag import ingest_chunks_to_neo4j

            chunk_ids = [f"chunk_{parent_id}_{i}" for i in range(len(chunks))]
            n_triples = ingest_chunks_to_neo4j(chunks, chunk_ids, safe_filename)
            task_state["graph_triples_ingested"] = n_triples
        except Exception as e:
            logger.warning("Neo4j graph ingest skipped: %s", e)

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

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "作为一个严谨的文档审核员，你需要判断以下【所有提供的内容段落】加在一起，是否包含了解答问题的关键事实依据？如果有哪怕一段相关，请回答 'yes'；如果全都是无关的废话，请坚决回答 'no'。"),
    ("human", "问题: {question}\n\n合并的上下文段落:\n{documents}")
])

GEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Using the information contained in the context, give a comprehensive answer to the question. "
        "Respond only to the question asked; the response should be concise and relevant to the question. "
        "Provide the number of the source document when relevant. "
        "If the answer cannot be deduced from the context, do not give an answer.",
    ),
    (
        "human",
        "Context:\n{context}\n"
        "---\n"
        "Now here is the question you need to answer.\n\n"
        "Question: {question}",
    ),
])

# 分析推理型问题使用宽松生成 Prompt：允许在文档事实基础上做专业推理
GEN_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一位严谨的专业分析师。请基于【知识库事实】，结合专业领域知识对问题进行深入分析。\n\n"
        "知识库事实（检索到的上下文）:\n{context}\n\n"
        "回答规范：\n"
        "1. 先陈述上下文中与问题直接相关的事实（标注为【文档事实】）\n"
        "2. 再基于这些事实运用专业知识进行分析推理（标注为【分析推理】）\n"
        "3. 分析推理部分可运用行业知识，但不得凭空捏造上下文中不存在的具体数字或人名"
    )),
    ("human", "问题: {question}")
])

# 分析型查询对应的宽松 grader prompt：只判断上下文是否含有主题相关事实
GRADE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一位文档相关性评估员。判断以下【上下文段落】是否包含与问题主题相关的背景事实，"
        "这些事实可作为进一步分析推理的基础。\n"
        "注意：不需要上下文直接回答问题，只需判断是否有该主题的相关背景信息。\n"
        "如果有任何主题相关事实，回答 'yes'；如果完全不相关，回答 'no'。"
    )),
    ("human", "问题: {question}\n\n上下文段落:\n{documents}")
])

_ANALYSIS_INTENT_KEYWORDS = frozenset([
    "分析", "影响", "推断", "预测", "建议", "方案", "规划", "策略",
    "如何", "为什么", "原因", "趋势", "对比", "评估", "风险", "机会",
    "挑战", "可行性", "展望", "下一步", "优化", "改进", "判断",
    "analyze", "analysis", "predict", "suggest", "plan", "why", "how",
    "impact", "feasibility", "risk", "opportunity", "challenge",
])

def _is_analysis_query(question: str) -> bool:
    """
    识别「分析推理型查询」：用户期望基于检索到的事实 + 专业知识做推理分析。
    与「事实型查询」的区别：不要求上下文有现成答案，只需有相关背景事实即可。
    """
    q = question.lower()
    return any(k in q for k in _ANALYSIS_INTENT_KEYWORDS)

def _is_analysis_followup(question: str) -> bool:
    """
    识别多轮对话中「基于已知项目做下一步分析/规划」的追问（带会话上下文信号）。
    """
    q = question.lower()
    has_memory_signal = (
        "topic anchor" in q
        or "conversation history" in q
        or "current user question" in q
    )
    return _is_analysis_query(q) and has_memory_signal


def retrieve_parent_documents_expanded(question: str, dense_k: int = 10) -> tuple[List[Document], List[str]]:
    """
    Vector retrieval over child chunks, expand to parent documents (Small-to-Big).
    Returns (expanded parent Documents, log lines). Used by vector-only path and Hybrid RAG.
    """
    logs: List[str] = []
    if not vectorstore:
        return [], ["Vector store not initialized."]
    logs.append(f"Dense retrieval: Top {dense_k} child chunks")
    retriever = vectorstore.as_retriever(search_kwargs={"k": dense_k})
    retrieved_chunks = retriever.invoke(question)
    logs.append("Context expansion: mapping chunks to parent documents")

    seen_parent_ids = set()
    parent_ids: List[str] = []
    for chunk in retrieved_chunks:
        pid = chunk.metadata.get("parent_id")
        if pid and pid not in seen_parent_ids:
            seen_parent_ids.add(pid)
            parent_ids.append(pid)

    parent_meta_by_id: Dict[str, dict] = {}
    batch_ok = False
    if parent_ids:
        try:
            for rec in collection.find({"type": "parent", "id": {"$in": parent_ids}}):
                pid = rec.get("id")
                if pid:
                    parent_meta_by_id[pid] = rec
            batch_ok = True
        except Exception as e:
            logger.error(f"Batch parent fetch failed: {e}")

    expanded_docs: List[Document] = []
    expanded_seen = set()
    for chunk in retrieved_chunks:
        pid = chunk.metadata.get("parent_id")
        if not pid or pid in expanded_seen:
            continue
        expanded_seen.add(pid)
        parent_record = parent_meta_by_id.get(pid) if batch_ok else collection.find_one({"id": pid, "type": "parent"})
        if parent_record:
            full_text = f"[Source: {parent_record.get('source', 'Unknown')}]\n{parent_record.get('full_content', '')}"
            expanded_docs.append(Document(page_content=full_text, metadata={"parent_id": pid, "source": "dense"}))

    if not expanded_docs:
        expanded_docs = retrieved_chunks
    return expanded_docs, logs


def retrieve_and_rerank_node(state: AgentState):
    logs = state.setdefault("logs", [])
    logs.append("Executing Vector Search (Top 10 child chunks)")
    if not vectorstore:
        return {"documents": [], "logs": logs}

    expanded_docs, sublogs = retrieve_parent_documents_expanded(state["question"], dense_k=10)
    logs.extend(sublogs)

    logs.append(f"Executing Jina Cross-Encoder Rerank (Top 3) on {len(expanded_docs)} Parent Documents")
    reranked_docs = jina_rerank(state["question"], expanded_docs, top_n=3)

    return {"documents": [doc.page_content for doc in reranked_docs], "logs": logs}

def grade_documents_node(state: AgentState):
    logs = state.setdefault("logs", [])
    logs.append("Grading context to prevent hallucination...")

    if not state["documents"]:
        return {"documents": [], "logs": logs}

    question = state.get("question", "")
    is_analysis = _is_analysis_query(question)
    combined_docs = "\n---\n".join(state["documents"])

    # 分析型查询使用宽松 grader，只判断是否有相关背景事实
    grade_prompt = GRADE_ANALYSIS_PROMPT if is_analysis else GRADE_PROMPT
    if is_analysis:
        logs.append("ℹ️ Analysis query detected: using relaxed grader.")

    try:
        result = grader_llm.invoke(grade_prompt.format_messages(
            question=question,
            documents=combined_docs
        ))
        if result and getattr(result, "binary_score", "").lower() == "yes":
            filtered_docs = state["documents"]
            logs.append("✅ Contexts approved for generation.")
        else:
            # 多轮追问兜底：带会话信号的分析型追问，有上下文时允许继续
            if state["documents"] and _is_analysis_followup(question):
                filtered_docs = state["documents"]
                logs.append("⚠️ Grader returned NO, but analysis-followup fallback enabled.")
            else:
                filtered_docs = []
                logs.append("⚠️ All contexts rejected. Hallucination blocked.")
    except Exception as e:
        logger.error(f"Grader failed: {e}", exc_info=True)
        filtered_docs = []
        logs.append("⚠️ Grader error; hallucination blocked (fail-closed).")

    return {"documents": filtered_docs, "logs": logs}

def generate_node(state: AgentState):
    if not state["documents"]:
        return {"generation": "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。"}

    logs = state.setdefault("logs", [])
    question = state.get("question", "")

    # 分析型查询使用允许推理的宽松 Prompt
    if _is_analysis_query(question):
        logs.append("ℹ️ Analysis query: using GEN_ANALYSIS_PROMPT.")
        prompt = GEN_ANALYSIS_PROMPT
    else:
        prompt = GEN_PROMPT

    logs.append("Generating answer based on verified context...")
    generation = llm.invoke(prompt.format_messages(
        context="\n".join(state["documents"]),
        question=question
    )).content
    return {"generation": generation, "logs": logs}


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_and_rerank_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)
rag_app = workflow.compile()


def langgraph_stream_log_enabled() -> bool:
    return os.environ.get("LANGGRAPH_STREAM_LOG", "").lower() in ("1", "true", "yes")


def invoke_rag_app(inputs: AgentState) -> AgentState:
    """
    Run the vector-only LangGraph. When LANGGRAPH_STREAM_LOG=true, uses ``stream(stream_mode="values")``
    so each node's state is logged (see LangGraph streaming docs); otherwise ``invoke``.
    """
    if not langgraph_stream_log_enabled():
        return rag_app.invoke(inputs)

    final_state: AgentState | None = None
    try:
        for step_idx, state in enumerate(rag_app.stream(inputs, stream_mode="values")):
            if not isinstance(state, dict):
                logger.info("[LangGraph stream] step=%s raw=%r", step_idx, state)
                continue
            final_state = state  # type: ignore[assignment]
            docs = state.get("documents") or []
            n_docs = len(docs) if isinstance(docs, list) else 0
            gen = state.get("generation") or ""
            logs = state.get("logs") or []
            n_log = len(logs) if isinstance(logs, list) else 0
            logger.info(
                "[LangGraph stream] step=%s | documents=%s | generation_chars=%s | log_lines=%s",
                step_idx,
                n_docs,
                len(gen) if isinstance(gen, str) else 0,
                n_log,
            )
    except Exception:
        logger.exception("LangGraph stream failed; falling back to invoke().")
        return rag_app.invoke(inputs)

    if final_state is None:
        logger.warning("[LangGraph stream] no chunks; falling back to invoke().")
        return rag_app.invoke(inputs)
    return final_state


def run_chat_pipeline(query: str):
    """
    Default: Hybrid RAG (router + dense/BM25 + Neo4j + fusion rerank) when HYBRID_RAG_ENABLED=true.
    Fallback: original LangGraph vector-only pipeline on failure or HYBRID_RAG_ENABLED=false.
    """
    if os.environ.get("HYBRID_RAG_ENABLED", "true").lower() in ("1", "true", "yes"):
        try:
            from src.hybrid_rag import run_hybrid_chat_pipeline

            return run_hybrid_chat_pipeline(query)
        except Exception:
            logger.exception("Hybrid pipeline failed; falling back to vector-only LangGraph.")
    inputs = {"question": query, "loop_count": 0, "logs": []}
    result = invoke_rag_app(inputs)
    return {
        "answer": result["generation"],
        "context_used": result["documents"],
        "logs": result["logs"],
    }

def reset_knowledge_base() -> None:
    """
    Clear MongoMock child/parent records and drop the LanceDB table; reset vectorstore handle.
    Intended for benchmarks / isolated tests. Set LANCEDB_URI before importing rag_core for a fresh path.
    """
    global vectorstore
    vectorstore = None
    try:
        collection.delete_many({})
    except Exception as e:
        logger.warning("reset_knowledge_base: mongo clear failed: %s", e)
    try:
        if TABLE_NAME in db_lance.table_names():
            db_lance.drop_table(TABLE_NAME)
    except Exception as e:
        logger.warning("reset_knowledge_base: lance drop failed: %s", e)


# --- Evaluation Logic ---
def retrieve_vector_ranked_documents(query: str, top_n: int = 10) -> List[Document]:
    """Vector-only path: dense Small-to-Big + Jina rerank (for Recall / NDCG)."""
    if not vectorstore:
        return []
    expanded_docs, _ = retrieve_parent_documents_expanded(query, dense_k=10)
    if not expanded_docs:
        return []
    return jina_rerank(query, expanded_docs, top_n=top_n)


def run_evaluation(query: str, expected_substring: str, use_hybrid: bool = False):
    """
    Binary relevance: first ranked document containing `expected_substring` wins.
    Reports Recall@1,3,5,10 and NDCG@K for K in 1,3,5,10 (primary: ndcg@10). See `src/recall_metrics.py`.
    """
    from src.recall_metrics import find_hit_rank, metrics_for_hit

    if not vectorstore:
        return {"error": "Database is empty."}

    if use_hybrid:
        from src.hybrid_rag import retrieve_hybrid_ranked_documents

        ranked_docs, meta = retrieve_hybrid_ranked_documents(query, top_n=10)
        if meta.get("error"):
            return {"error": meta["error"], **{k: v for k, v in meta.items() if k != "logs"}}
        ranked_texts = [d.page_content for d in ranked_docs]
        eval_logs = list(meta.get("logs", []))
        route_info = {
            "route": meta.get("route"),
            "effective_route": meta.get("effective_route"),
            "router_reason": meta.get("router_reason"),
        }
    else:
        ranked_docs = retrieve_vector_ranked_documents(query, top_n=10)
        ranked_texts = [d.page_content for d in ranked_docs]
        eval_logs = ["Vector-only eval: dense + small-to-big + Jina rerank top 10."]
        route_info = {}

    hit_rank = find_hit_rank(ranked_texts, expected_substring)
    per = metrics_for_hit(hit_rank)

    legacy: Dict[str, Dict[str, float | int]] = {}
    for k in (1, 3, 5, 10):
        legacy[f"K={k}"] = {
            "Recall": per[f"recall@{k}"],
            "NDCG": per[f"ndcg@{k}"],
        }

    out: Dict[str, Any] = {
        "query": query,
        "expected_substring": expected_substring,
        "use_hybrid": use_hybrid,
        "hit_rank": hit_rank + 1 if hit_rank != -1 else "Not Found",
        "hit_rank_0": hit_rank,
        "metrics": per,
        "metrics_legacy": legacy,
        "top_3_docs": [d.page_content for d in ranked_docs[:3]],
        "eval_logs": eval_logs,
    }
    out.update(route_info)
    return out

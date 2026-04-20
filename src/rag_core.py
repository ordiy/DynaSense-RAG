"""
RAG core: LangGraph pipeline, embeddings, and vector storage.

Vector + parent/child text live in **PostgreSQL** (JSONB ``kb_doc`` + pgvector ``kb_embedding``).
See ``docs/postgresql_storage_roadmap.md``.

LangSmith: call ``src.observability.init_langsmith_tracing()`` before this module
is imported (``app.py`` does this for the API server). Standalone scripts should
call it first if tracing is desired.
"""
import os
import json
import time
import uuid
import requests
from typing import Any, Dict, Iterator, List, Tuple, TypedDict
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

from src.core.config import get_settings
from src.core.rag_context_format import format_numbered_passages

import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")

# Jina 外部调用：超时/重试，避免 worker 长时间阻塞
JINA_REQUEST_TIMEOUT = (5, 20)  # (connect_timeout, read_timeout)
JINA_MAX_RETRIES = 3

# --- Embeddings / LLM (shared by all storage backends) ---
doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)

# Populated by ``_setup_storage()`` — PostgreSQL only.
collection = None  # PostgresJsonbDocCollection (JSONB ``kb_doc``)
vectorstore = None  # PostgresVectorStore


def _init_postgres_storage() -> None:
    """PostgreSQL: pgvector rows + JSONB parent/child docs for BM25 / small-to-big."""
    global collection, vectorstore
    from src.infrastructure.persistence.postgres_jsonb_collection import PostgresJsonbDocCollection
    from src.infrastructure.persistence.postgres_connection import get_pool, init_pool
    from src.infrastructure.persistence.postgres_schema import ensure_schema as pg_ensure_schema
    from src.infrastructure.persistence.postgres_vectorstore import PostgresVectorStore

    s = get_settings()
    url = s.database_url
    if not url:
        raise RuntimeError(
            "DATABASE_URL is required (PostgreSQL + pgvector). "
            "Example: postgresql://user:pass@localhost:5432/map_rag"
        )
    init_pool(url)
    pg_ensure_schema(get_pool())
    collection = PostgresJsonbDocCollection(get_pool())
    vectorstore = PostgresVectorStore(get_pool(), doc_embeddings)


def setup_storage() -> None:
    """
    Initialize unified PostgreSQL storage.

    Called once from the FastAPI lifespan handler (``src/api/main.py``).
    Safe to call multiple times (``init_pool`` is idempotent).
    Skipped silently when ``DATABASE_URL`` is absent so the module can be
    imported in unit-test environments without a live database.
    """
    from src.core.config import get_settings

    if not get_settings().database_url:
        logger.warning(
            "DATABASE_URL not set — RAG storage not initialised. "
            "Vector/graph operations will be unavailable."
        )
        return
    _init_postgres_storage()
    logger.info("RAG storage: PostgreSQL + pgvector (unified).")

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

# Vertex AI embedding API limit: max 250 texts per batch call.
_EMBED_BATCH_SIZE = 250


def _embed_in_batches(texts: list[str]) -> list:
    """Call embed_documents in ≤250-item batches to stay within Vertex AI limits."""
    all_embeddings = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i: i + _EMBED_BATCH_SIZE]
        all_embeddings.extend(doc_embeddings.embed_documents(batch))
    return all_embeddings


# --- Ingestion Task ---
def process_document_task(content: str, filename: str, task_state: dict):
    global vectorstore
    try:
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

        if mongo_records:
            collection.insert_many(mongo_records)

        task_state["progress"] = 70

        # Vector index: batch embed to respect the 250-instance-per-request API limit.
        if docs_to_insert:
            texts = [d.page_content for d in docs_to_insert]
            embeds = _embed_in_batches(texts)
            rows_pg = [
                (d.metadata["doc_id"], d.page_content, dict(d.metadata), embeds[j])
                for j, d in enumerate(docs_to_insert)
            ]
            assert vectorstore is not None
            vectorstore.add_embedding_rows(rows_pg)

        # Graph: LLM triple extraction + merge (optional; skip for benchmarks)
        if not get_settings().skip_graph_ingest:
            try:
                from src.hybrid_rag import ingest_chunks_to_graph

                chunk_ids = [f"chunk_{parent_id}_{i}" for i in range(len(chunks))]
                n_triples = ingest_chunks_to_graph(chunks, chunk_ids, filename)
                task_state["graph_triples_ingested"] = n_triples
            except Exception as e:
                logger.warning("Graph ingest skipped: %s", e)

        task_state["status"] = "completed"
        task_state["progress"] = 100
        task_state["result"] = f"Processed {len(chunks)} chunks."

    except Exception as exc:
        logger.exception("process_document_task failed for %s: %s", filename, exc)
        task_state["status"] = "failed"
        task_state["progress"] = 0
        task_state["error"] = str(exc)

# --- Retrieval Pipeline (LangGraph) ---
class AgentState(TypedDict, total=False):
    question: str
    expanded_questions: List[str]
    documents: List[str]
    generation: str
    loop_count: int
    logs: List[str]
    # Set once in retrieve_and_rerank_node; reused by grade and generate nodes.
    is_analysis: bool
    # Set to True by the hybrid path to skip vector retrieval (docs already provided).
    skip_retrieval: bool

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes or no")

grader_llm = llm.with_structured_output(GradeDocuments)

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "作为一个严谨的文档审核员，用户问题下方会给出若干段 **按编号排列的检索结果**（[Passage 1]、[Passage 2]…）。"
        "你需要通读每一段，判断这些段落 **合起来** 是否包含解答问题所需的关键事实依据。"
        "交叉编码器重排后，排名靠后的段落仍可能包含互补信息，请不要只根据第一段下结论。"
        "只要有任一段落与问题实质相关，请回答 'yes'；若全部与问题无关，请回答 'no'。",
    ),
    ("human", "问题: {question}\n\n编号检索段落:\n{documents}"),
])

GEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are given a question and multiple numbered passages (cross-encoder reranked). "
        "Lead with the direct answer in 1–2 sentences before elaborating — this ensures the "
        "key finding survives any downstream truncation. "
        "Integrate evidence from every passage that materially helps answer the question; "
        "do not default to only the first passage unless later passages add nothing. "
        "When multiple passages support different aspects, synthesize them. "
        "If passages conflict, state the uncertainty and which passage(s) support each view. "
        "Respond only to the question; be concise. Reference passage numbers when citing (e.g. Passage 2). "
        "If the answer cannot be deduced from the passages, do not answer.",
    ),
    (
        "human",
        "Context (numbered passages):\n{context}\n"
        "---\n"
        "Question: {question}",
    ),
])

# 分析推理型问题使用宽松生成 Prompt：允许在文档事实基础上做专业推理
GEN_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一位严谨的专业分析师。下面是以 [Passage 1]…[Passage N] 编号的检索结果（已重排）；"
        "请先用1–2句话直接回答问题，再展开引用与分析——这样即使回答被截断，核心结论依然保留。\n"
        "综合 **所有与主题相关的段落**，不要只使用最靠前的一段，除非其余段落确实无关。"
        "知识库事实（检索到的上下文）:\n{context}\n\n"
        "回答规范：\n"
        "1. 先分点陈述各编号段落中与问题相关的事实（标注为【文档事实】，可引用 Passage 编号）\n"
        "2. 再基于这些事实运用专业知识进行分析推理（标注为【分析推理】）\n"
        "3. 分析推理可运用行业知识，但不得凭空捏造上下文中不存在的具体数字或人名；若段落冲突，说明分歧与依据"
    )),
    ("human", "问题: {question}")
])

# 分析型查询对应的宽松 grader prompt：只判断上下文是否含有主题相关事实
GRADE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一位文档相关性评估员。下方为按编号排列的检索段落（[Passage 1]…）。"
        "请通读各段，判断整体上是否包含与问题主题相关的背景事实（可作分析推理的基础）。"
        "排名靠后的段落也可能含有关键补充信息，请一并考虑。\n"
        "不需要上下文直接回答问题，只需判断是否存在主题相关背景。\n"
        "若有任何相关事实，回答 'yes'；若全部与主题无关，回答 'no'。"
    )),
    ("human", "问题: {question}\n\n编号上下文段落:\n{documents}")
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


class ExpandQuery(BaseModel):
    queries: List[str] = Field(description="List of exactly 2 rephrased questions")

expand_llm = llm.with_structured_output(ExpandQuery)

EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert search query expander. Given the user's question, generate exactly 2 distinct rephrased versions of the question that use different synonyms or terminology to maximize retrieval recall. Output ONLY the two strings in the required format."
    ),
    ("human", "{question}")
])

def expand_query_node(state: AgentState):
    logs = state.setdefault("logs", [])
    question = state["question"]
    if not get_settings().query_expansion_enabled:
        return {"expanded_questions": [question], "logs": logs}
    
    logs.append("Expanding query...")
    try:
        result = expand_llm.invoke(EXPAND_PROMPT.format_messages(question=question))
        rephrased = getattr(result, "queries", [])
        expanded = [question] + list(rephrased)[:2]
        logs.append(f"Query expanded to {len(expanded)} variants.")
        return {"expanded_questions": expanded, "logs": logs}
    except Exception as e:
        logger.error(f"Query expansion failed: {e}", exc_info=True)
        return {"expanded_questions": [question], "logs": logs + ["⚠️ Query expansion failed, using original query."]}


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
    if parent_ids:
        try:
            for rec in collection.find({"type": "parent", "id": {"$in": parent_ids}}):
                pid = rec.get("id")
                if pid:
                    parent_meta_by_id[pid] = rec
        except Exception as e:
            logger.error("Batch parent fetch failed: %s", e)
            # Return child chunks as fallback rather than making N individual DB calls.
            return retrieved_chunks, logs + ["Parent batch fetch failed; returning child chunks."]

    expanded_docs: List[Document] = []
    expanded_seen = set()
    for chunk in retrieved_chunks:
        pid = chunk.metadata.get("parent_id")
        if not pid or pid in expanded_seen:
            continue
        expanded_seen.add(pid)
        parent_record = parent_meta_by_id.get(pid)
        if parent_record:
            full_text = f"[Source: {parent_record.get('source', 'Unknown')}]\n{parent_record.get('full_content', '')}"
            expanded_docs.append(Document(page_content=full_text, metadata={"parent_id": pid, "source": "dense"}))

    if not expanded_docs:
        expanded_docs = retrieved_chunks
    return expanded_docs, logs


def retrieve_and_rerank_node(state: AgentState):
    logs = state.setdefault("logs", [])
    question = state["question"]
    # Compute is_analysis once; reused by grade_documents_node and generate_node.
    is_analysis = _is_analysis_query(question)

    # Hybrid path pre-populates documents and sets skip_retrieval=True.
    if state.get("skip_retrieval"):
        logs.append("retrieve_and_rerank_node: skip (documents pre-populated by hybrid path).")
        return {"is_analysis": is_analysis, "logs": logs}

    logs.append("Executing Vector Search (Top 10 child chunks)")
    if not vectorstore:
        return {"documents": [], "is_analysis": is_analysis, "logs": logs}

    top_n = get_settings().rag_vector_rerank_top_n

    expanded_qs = state.get("expanded_questions")
    if not expanded_qs:
        expanded_qs = [question]

    all_expanded_docs = []
    seen_chunk_ids = set()
    from src.core.query_anchors import filter_documents_by_query_anchors

    for q in expanded_qs:
        expanded_docs, sublogs = retrieve_parent_documents_expanded(q, dense_k=10)
        logs.extend(sublogs)
        expanded_docs, alogs = filter_documents_by_query_anchors(q, expanded_docs)
        logs.extend(alogs)
        
        for doc in expanded_docs:
            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id") or doc.page_content
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                all_expanded_docs.append(doc)

    logs.append(
        f"Executing Jina Cross-Encoder Rerank (top_n={top_n}) on {len(all_expanded_docs)} parent documents"
    )
    reranked_docs = jina_rerank(question, all_expanded_docs, top_n=top_n)

    s = get_settings()
    if s.mmr_enabled and len(reranked_docs) > 1:
        from src.core.mmr import mmr_filter
        reranked_docs = mmr_filter(reranked_docs, k=top_n, lambda_param=s.mmr_lambda)
        logs.append(f"MMR applied: {len(reranked_docs)} docs (lambda={s.mmr_lambda})")

    return {"documents": [doc.page_content for doc in reranked_docs], "is_analysis": is_analysis, "logs": logs}

def grade_documents_node(state: AgentState):
    logs = state.setdefault("logs", [])
    logs.append("Grading context to prevent hallucination...")

    if not state.get("documents"):
        return {"documents": [], "logs": logs}

    question = state.get("question", "")
    # Reuse the flag computed once in retrieve_and_rerank_node; fall back to direct check
    # when state was built outside the LangGraph path (e.g. tests).
    is_analysis = state["is_analysis"] if "is_analysis" in state else _is_analysis_query(question)
    combined_docs = format_numbered_passages(state["documents"])

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
    if not state.get("documents"):
        return {"generation": "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。"}

    logs = state.setdefault("logs", [])
    question = state.get("question", "")

    # Reuse flag computed once in retrieve_and_rerank_node.
    is_analysis = state["is_analysis"] if "is_analysis" in state else _is_analysis_query(question)
    if is_analysis:
        logs.append("ℹ️ Analysis query: using GEN_ANALYSIS_PROMPT.")
        prompt = GEN_ANALYSIS_PROMPT
    else:
        prompt = GEN_PROMPT

    logs.append("Generating answer based on verified context...")
    ctx = format_numbered_passages(state["documents"])
    generation = llm.invoke(prompt.format_messages(
        context=ctx,
        question=question
    )).content
    return {"generation": generation, "logs": logs}


def stream_generation_chunks(
    question: str,
    document_strings: List[str],
    is_analysis: bool | None = None,
) -> Iterator[str]:
    """
    Stream LLM output with the same prompt choice as ``generate_node`` (Vertex ``llm.stream``).

    ``is_analysis`` can be passed in from a caller that already computed it (e.g.
    ``iter_hybrid_chat_stream_events``) to avoid a redundant keyword scan.
    When ``None``, it is derived from ``question`` directly.

    Yields text fragments; concatenation matches non-streaming generation for the same context.
    """
    if not document_strings:
        yield "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。"
        return
    # Match ``generate_node`` formatting so streaming and non-stream outputs align.
    context = format_numbered_passages(document_strings)
    if is_analysis is None:
        is_analysis = _is_analysis_query(question)
    if is_analysis:
        msgs = GEN_ANALYSIS_PROMPT.format_messages(context=context, question=question)
    else:
        msgs = GEN_PROMPT.format_messages(context=context, question=question)
    for chunk in llm.stream(msgs):
        text = getattr(chunk, "content", None)
        if isinstance(text, str) and text:
            yield text
        elif isinstance(text, list):
            for part in text:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text") or ""
                    if t:
                        yield t


def iter_vector_chat_stream_events(question: str) -> Iterator[Dict[str, Any]]:
    """
    Vector-only path: retrieve → rerank → grade → stream tokens. Used when hybrid is off or fails.
    """
    from src.core.citations import build_citations_from_context

    inputs: AgentState = {"question": question, "loop_count": 0, "logs": []}
    inputs.update(expand_query_node(inputs))
    inputs.update(retrieve_and_rerank_node(inputs))
    inputs.update(grade_documents_node(inputs))
    docs = inputs.get("documents") or []
    citations = build_citations_from_context(docs)
    yield {"type": "meta", "citations": citations, "logs": inputs.get("logs") or []}
    blocked = "抱歉，知识库中未能找到与您问题高度相关的信息，为避免产生误导（幻觉），系统已阻断回答。"
    if not docs:
        yield {"type": "done", "answer": blocked}
        return
    parts: List[str] = []
    for piece in stream_generation_chunks(question, docs, is_analysis=inputs.get("is_analysis")):
        parts.append(piece)
        yield {"type": "token", "text": piece}
    yield {"type": "done", "answer": "".join(parts)}


def iter_chat_stream_events(question: str) -> Iterator[Dict[str, Any]]:
    """
    Product streaming API: dispatch hybrid vs vector, mirroring ``run_chat_pipeline``.
    """
    if get_settings().hybrid_rag_enabled:
        try:
            from src.hybrid_rag import iter_hybrid_chat_stream_events

            yield from iter_hybrid_chat_stream_events(question)
            return
        except Exception:
            logger.exception("Hybrid stream failed; falling back to vector-only stream.")
    yield from iter_vector_chat_stream_events(question)


workflow = StateGraph(AgentState)
workflow.add_node("expand_query", expand_query_node)
workflow.add_node("retrieve", retrieve_and_rerank_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "expand_query")
workflow.add_edge("expand_query", "retrieve")
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
    Dispatch order:
      1. Agentic ReAct loop (``agentic_retrieval_enabled=True``) — multi-hop, highest recall.
      2. Hybrid RAG (``hybrid_rag_enabled=True``) — fast, production default.
      3. Vector-only LangGraph — fallback.
    """
    from src.core.citations import build_citations_from_context

    s = get_settings()
    if s.agentic_retrieval_enabled:
        try:
            from src.agentic_rag import run_agentic_chat_pipeline

            return run_agentic_chat_pipeline(query)
        except Exception:
            logger.exception("Agentic pipeline failed; falling back to hybrid/vector.")

    if get_settings().hybrid_rag_enabled:
        try:
            from src.hybrid_rag import run_hybrid_chat_pipeline

            out = run_hybrid_chat_pipeline(query)
            out["citations"] = build_citations_from_context(out.get("context_used"))
            return out
        except Exception:
            logger.exception("Hybrid pipeline failed; falling back to vector-only LangGraph.")
    inputs = {"question": query, "loop_count": 0, "logs": []}
    result = invoke_rag_app(inputs)
    ctx = result["documents"]
    return {
        "answer": result["generation"],
        "context_used": ctx,
        "logs": result["logs"],
        "citations": build_citations_from_context(ctx),
    }

def run_chat_pipeline_multimodal(
    question: str,
    image_parts: List[Tuple[str, str]],  # [(mime_type, base64_data), ...]
) -> dict:
    """
    Multimodal RAG pipeline: vector retrieval + Gemini vision.
    Called when the user attaches one or more images to the chat message.
    Text documents attached by the user should be extracted and prepended to
    ``question`` by the caller before invoking this function.
    """
    from langchain_core.messages import HumanMessage
    from src.core.citations import build_citations_from_context

    logs: List[str] = []
    context_str = ""
    context_texts: List[str] = []

    # Vector retrieval – only when the KB is initialised and there is a text query.
    if vectorstore and question.strip():
        top_n = get_settings().rag_vector_rerank_top_n
        expanded_docs, sublogs = retrieve_parent_documents_expanded(question, dense_k=10)
        logs.extend(sublogs)
        reranked = jina_rerank(question, expanded_docs, top_n=top_n)
        context_texts = [d.page_content for d in reranked]
        context_str = format_numbered_passages(context_texts)
        logs.append(f"Multimodal: retrieved {len(context_texts)} context passage(s).")
    else:
        logs.append("Multimodal: no KB context (vector store not ready or empty query).")

    # Build multimodal LangChain message: images first, then text.
    content_parts: list = []
    for mime_type, b64_data in image_parts:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
        })

    text_prompt = question
    if context_str:
        text_prompt = (
            f"Knowledge base context (numbered passages):\n{context_str}\n\n"
            f"---\n"
            f"Question: {question}"
        )
    content_parts.append({"type": "text", "text": text_prompt})

    response = llm.invoke([HumanMessage(content=content_parts)])
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "logs": logs,
        "context_used": context_texts,
        "citations": build_citations_from_context(context_texts),
        "route": "multimodal",
    }


def reset_knowledge_base() -> None:
    """
    Clear all indexed documents (PostgreSQL TRUNCATE via ``collection.delete_many``).

    Intended for benchmarks / isolated tests. Requires ``DATABASE_URL``.
    """
    global vectorstore
    try:
        collection.delete_many({})
    except Exception as e:
        logger.warning("reset_knowledge_base: truncate failed: %s", e)
    try:
        from src.infrastructure.persistence.postgres_vectorstore import PostgresVectorStore
        from src.infrastructure.persistence.postgres_connection import get_pool

        vectorstore = PostgresVectorStore(get_pool(), doc_embeddings)
    except Exception as e:
        logger.warning("reset_knowledge_base: vectorstore refresh failed: %s", e)


# --- Evaluation Logic ---
def retrieve_vector_ranked_documents(query: str, top_n: int = 10) -> List[Document]:
    """Vector-only path: dense Small-to-Big + optional anchor filter + Jina rerank (for Recall / NDCG)."""
    if not vectorstore:
        return []
    expanded_docs, _ = retrieve_parent_documents_expanded(query, dense_k=10)
    if not expanded_docs:
        return []
    from src.core.query_anchors import filter_documents_by_query_anchors

    expanded_docs, _ = filter_documents_by_query_anchors(query, expanded_docs)
    if not expanded_docs:
        return []
    return jina_rerank(query, expanded_docs, top_n=top_n)


def run_evaluation(
    query: str,
    expected_substring: str,
    use_hybrid: bool = False,
    compute_faithfulness: bool = False,
):
    """
    Dual-metric evaluation: retrieval quality + optional generation faithfulness.

    Retrieval metrics (always): Recall@1,3,5,10 and NDCG@K (see ``src/recall_metrics.py``).

    Faithfulness (opt-in, ``compute_faithfulness=True``):
      Runs the generation step on the retrieved passages, then calls
      ``judge_faithfulness`` (LLM-as-a-Judge). Adds ~2–5 s per query.
      Returns ``faithfulness_score`` (0.0–1.0), ``faithfulness_verdict``,
      ``faithfulness_reasoning`` in the result dict.

    Motivation: Recall@K only tells us whether the right document was retrieved;
    faithfulness tells us whether the generated answer is grounded in that document.
    These two metrics together close the measurement gap surfaced by DoTA-RAG
    (arXiv 2506.12571): a system can have high Recall@K but low Faithfulness when
    the generator extrapolates beyond the retrieved context.
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

    if compute_faithfulness and ranked_texts:
        try:
            from src.core.faithfulness import judge_faithfulness

            # Generate an answer from the ranked passages for faithfulness auditing.
            is_analysis = _is_analysis_query(query)
            ctx = format_numbered_passages(ranked_texts)
            prompt = GEN_ANALYSIS_PROMPT if is_analysis else GEN_PROMPT
            answer = llm.invoke(
                prompt.format_messages(context=ctx, question=query)
            ).content
            eval_logs.append(f"Faithfulness: generated answer ({len(answer)} chars) for auditing.")

            faith = judge_faithfulness(answer, ranked_texts, llm)
            out["answer"] = answer
            out["metrics"]["faithfulness_score"] = faith["faithfulness_score"]
            out["faithfulness_verdict"] = faith["faithfulness_verdict"]
            out["faithfulness_reasoning"] = faith["faithfulness_reasoning"]
            eval_logs.append(
                f"Faithfulness verdict: {faith['faithfulness_verdict']} "
                f"(score={faith['faithfulness_score']})"
            )
        except Exception as exc:
            logger.warning("run_evaluation: faithfulness step failed: %s", exc)
            out["metrics"]["faithfulness_score"] = 0.0
            out["faithfulness_verdict"] = "judge_error"
            out["faithfulness_reasoning"] = str(exc)

    out["eval_logs"] = eval_logs
    return out

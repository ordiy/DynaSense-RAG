import os
import json
import requests
import lancedb
from typing import List, TypedDict
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
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
    # Using newlines to help jina segment properly based on our test
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

# --- Reranking with Mongomock (Simulating minimongo) ---
import mongomock

mongo_client = mongomock.MongoClient()
db_mongo = mongo_client.doc_db
collection = db_mongo.chunks

def index_chunks_to_mongo(docs: List[Document]):
    collection.delete_many({})
    records = []
    for doc in docs:
        records.append({
            "id": doc.metadata["id"],
            "parent_id": doc.metadata["parent_id"],
            "content": doc.page_content,
            # In a real scenario, you'd store dense/sparse vectors here too.
        })
    collection.insert_many(records)

def mock_rerank_mongo(query: str, retrieved_docs: List[Document], top_n: int = 2) -> List[Document]:
    """
    Simulates a rerank step by querying the document index (Mongo).
    In a real system, you'd use BM25 or Cross-Encoder here.
    Here we do a simple keyword frequency match to simulate reranking scoring.
    """
    if not retrieved_docs:
        return []
        
    scored_docs = []
    query_keywords = set(query.lower()) # Very naive "tokenizer"
    
    for doc in retrieved_docs:
        # Fetch original chunk from mongo to prove integration
        mongo_record = collection.find_one({"id": doc.metadata["id"]})
        content = mongo_record["content"] if mongo_record else doc.page_content
        
        # Simple simulated score: overlap of chars/words
        score = sum(1 for kw in query_keywords if kw in content.lower())
        scored_docs.append((score, doc))
        
    # Sort descending by score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_n]]

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
    chunks = chunk_text_jina(doc["text"])
    for chunk_text in chunks:
        chunked_documents.append(
            Document(
                page_content=chunk_text, 
                metadata={"id": chunk_id_counter, "parent_id": doc["id"]}
            )
        )
        chunk_id_counter += 1

print(f"Original docs: {len(raw_docs)}, Total chunks after Jina: {len(chunked_documents)}")

print("Indexing to MongoDB (for Rerank) and LanceDB (for Vector)...")
index_chunks_to_mongo(chunked_documents)

db_lance = lancedb.connect("/tmp/lancedb_mvp_v2")
table_name = "doc_chunks_v2"
if table_name in db_lance.table_names():
    db_lance.drop_table(table_name)

vectorstore = LanceDB.from_documents(
    documents=chunked_documents,
    embedding=doc_embeddings,
    connection=db_lance,
    table_name=table_name
)

# Fetch more initially for reranking
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LangGraph Definition ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    loop_count: int

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="文档是否与问题相关？回答 'yes' 或 'no'")

grader_llm = llm.with_structured_output(GradeDocuments)

def retrieve_and_rerank_node(state: AgentState):
    print("--- [感知层] 执行向量检索 (Recall) ---")
    retrieved_docs = retriever.invoke(state["question"])
    
    print("--- [感知层] 执行重排 (Rerank) ---")
    reranked_docs = mock_rerank_mongo(state["question"], retrieved_docs, top_n=2)
    
    return {"documents": [doc.page_content for doc in reranked_docs]}

def grade_documents_node(state: AgentState):
    print("--- [认知层] 评估感知质量 (防幻觉) ---")
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
        except Exception as e:
            print(f"Grader LLM failed: {e}")
            filtered_docs.append(doc)
    return {"documents": filtered_docs}

def rewrite_query_node(state: AgentState):
    print("--- [执行层] 检索失败，触发经验闭环，重写 Query ---")
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "意图优化专家：请根据原问题，换一种表达方式或提取核心关键词以用于更好的向量检索。直接输出新问题，不要其他解释。"),
        ("human", "原问题: {question}")
    ])
    new_question = llm.invoke(rewrite_prompt.format_messages(question=state["question"])).content
    return {"question": new_question, "loop_count": state.get("loop_count", 0) + 1}

def generate_node(state: AgentState):
    print("--- [执行层] 结合精选记忆生成回答 ---")
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "使用以下检索到的上下文来回答问题。\n上下文: {context}"),
        ("human", "问题: {question}")
    ])
    generation = llm.invoke(gen_prompt.format_messages(context="\n".join(state["documents"]), question=state["question"])).content
    return {"generation": generation}

def decide_to_generate(state: AgentState):
    if len(state["documents"]) > 0 or state.get("loop_count", 0) >= 2:
        return "generate"
    return "rewrite"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_and_rerank_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_to_generate, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app_v2 = workflow.compile()

# --- Evaluation ---
def run_evaluation_v2():
    print("\n========== v1.1 开始评估 Recall 指标 (Chunking + Rerank) ==========")
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
        # Get K=5 from vector
        initial_docs = vectorstore.similarity_search(t["query"], k=5)
        # Rerank to Top=2
        final_docs = mock_rerank_mongo(t["query"], initial_docs, top_n=2)
        
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
    
    # Generate reports
    report = {
        "version": "1.1-chunking-rerank",
        "metrics": {
            "total_queries": len(test_queries),
            "correct_hits": correct_retrievals,
            "recall_at_2": recall_rate
        },
        "details": results_data
    }
    
    with open("recall_metrics_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
        
    with open("recall_report_v2.md", "w", encoding="utf-8") as f:
        f.write("# MVP v1.1 Recall Metrics Report (Chunking + Rerank)\n\n")
        f.write(f"- **Total Queries Evaluated:** {len(test_queries)}\n")
        f.write(f"- **Recall@2 Rate:** {recall_rate*100:.2f}%\n\n")
        f.write("## Detailed Query Results\n\n")
        f.write("| Query | Expected Parent ID | Retrieved Parent IDs | Hit |\n")
        f.write("|---|---|---|---|\n")
        for d in results_data:
            hit_emoji = "✅" if d["hit"] else "❌"
            f.write(f"| {d['query']} | {d['expected_parent_id']} | {d['retrieved_parent_ids']} | {hit_emoji} |\n")
            
    print("Metrics data saved to recall_metrics_v2.json and recall_report_v2.md")

if __name__ == "__main__":
    run_evaluation_v2()

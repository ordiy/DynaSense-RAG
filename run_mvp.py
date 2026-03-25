import os
import json
import lancedb
from typing import List, TypedDict
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

# 1. Models and Embeddings Config
print("Initializing models...")
# Using Vertex AI real embeddings and the newly available gemini-2.5-pro
doc_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
query_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
llm = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0)

# 2. Database and Ingestion (LanceDB)
print("Initializing LanceDB and ingesting data...")
db = lancedb.connect("/tmp/lancedb_mvp")
table_name = "doc_features"

# Generate a synthetic dataset for testing recall
sample_docs = [
    Document(page_content="公司 2023 年 Q3 营收为 500 万美元，主要得益于 AI 产品的订阅增长。", metadata={"id": 1}),
    Document(page_content="2024 年战略规划：全面投入 Agentic 架构研发，特别是 MAP-RAG 系统。", metadata={"id": 2}),
    Document(page_content="我们的旗舰产品 'DataSphere' 用户量在上个月突破了 10 万大关。", metadata={"id": 3}),
    Document(page_content="人力资源部通知：本周五下午将举行年度团建活动，地点在奥林匹克公园。", metadata={"id": 4}),
    Document(page_content="技术预研：关于新一代 Matryoshka 表征学习在向量搜索中的降本增效分析。", metadata={"id": 5}),
    Document(page_content="客户反馈：近期有 15% 的用户抱怨新版本的加载速度变慢，需要前端团队重点关注。", metadata={"id": 6}),
    Document(page_content="财务审批流程更新：所有超过 1000 美元的采购都需要经过部门经理和 CFO 的双重审批。", metadata={"id": 7}),
    Document(page_content="开源社区贡献指南：我们鼓励员工在工作时间参与开源项目，并将其作为绩效考核的一部分。", metadata={"id": 8}),
    Document(page_content="服务器迁移计划：预计在下个月中旬，我们将所有的后端服务从 AWS 迁移到 Google Cloud。", metadata={"id": 9}),
    Document(page_content="新员工入职培训：每月的第一个星期一为新员工统一入职培训日，涵盖公司文化和基础安全操作。", metadata={"id": 10})
]

if table_name in db.table_names():
    db.drop_table(table_name)

vectorstore = LanceDB.from_documents(
    documents=sample_docs,
    embedding=doc_embeddings,
    connection=db,
    table_name=table_name
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. LangGraph Definition
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    loop_count: int

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="文档是否与问题相关？回答 'yes' 或 'no'")

grader_llm = llm.with_structured_output(GradeDocuments)

def retrieve_node(state: AgentState):
    print("--- [感知层] 执行向量检索 ---")
    docs = retriever.invoke(state["question"])
    return {"documents": [doc.page_content for doc in docs]}

def grade_documents_node(state: AgentState):
    print("---[认知层] 评估感知质量 (防幻觉) ---")
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
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_to_generate, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()

# 4. Evaluation and Recall testing
def run_evaluation():
    print("\n========== 开始评估 Recall 指标 ==========")
    test_queries = [
        {"query": "咱们公司未来打算搞什么技术架构？", "expected_id": 2},
        {"query": "公司23年三季度的营收情况", "expected_id": 1},
        {"query": "降维向量的相关的技术分析", "expected_id": 5},
        {"query": "星期五有什么安排？", "expected_id": 4},
        {"query": "前端新版本加载慢怎么回事？", "expected_id": 6},
        {"query": "入职培训什么时间？", "expected_id": 10}
    ]

    correct_retrievals = 0
    results_data = []
    print("Testing Recall@2 for raw retrieval...")
    
    for t in test_queries:
        docs = vectorstore.similarity_search(t["query"], k=2)
        retrieved_ids = [d.metadata["id"] for d in docs]
        is_hit = t["expected_id"] in retrieved_ids
        if is_hit:
            correct_retrievals += 1
            
        print(f"Query: '{t['query']}' -> Retrieved IDs: {retrieved_ids}, Expected: {t['expected_id']}, Hit: {is_hit}")
        
        results_data.append({
            "query": t["query"],
            "expected_id": t["expected_id"],
            "retrieved_ids": retrieved_ids,
            "hit": is_hit
        })

    recall_rate = correct_retrievals / len(test_queries)
    print(f"Recall@2: {correct_retrievals}/{len(test_queries)} ({recall_rate*100:.2f}%)")
    
    # Generate and save reports
    report = {
        "metrics": {
            "total_queries": len(test_queries),
            "correct_hits": correct_retrievals,
            "recall_at_2": recall_rate
        },
        "details": results_data
    }
    
    with open("recall_metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
        
    with open("recall_report.md", "w", encoding="utf-8") as f:
        f.write("# MVP Recall Metrics Report\n\n")
        f.write(f"- **Total Queries Evaluated:** {len(test_queries)}\n")
        f.write(f"- **Recall@2 Rate:** {recall_rate*100:.2f}%\n\n")
        f.write("## Detailed Query Results\n\n")
        f.write("| Query | Expected ID | Retrieved IDs | Hit |\n")
        f.write("|---|---|---|---|\n")
        for d in results_data:
            hit_emoji = "✅" if d["hit"] else "❌"
            f.write(f"| {d['query']} | {d['expected_id']} | {d['retrieved_ids']} | {hit_emoji} |\n")
            
    print("Metrics data saved to recall_metrics_report.json and recall_report.md")

# 5. Graph execution Test
def test_graph():
    print("\n========== 测试：LangGraph 闭环执行 (使用真实 Gemini 模型) ==========")
    test_query = "上个月破了十万，是哪个产品？"
    print(f"User Query: {test_query}")
    inputs = {"question": test_query, "loop_count": 0}
    
    result = app.invoke(inputs)
    print(f"\nFinal State Question: {result['question']}")
    print(f"Loop Count: {result.get('loop_count', 0)}")
    print(f"Documents Used: {result['documents']}")
    print(f"最终回答: {result['generation']}")

if __name__ == "__main__":
    run_evaluation()
    test_graph()

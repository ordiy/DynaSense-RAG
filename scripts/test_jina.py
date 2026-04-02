import os
import requests

key = os.environ.get("JINA_API_KEY", "")
if not key:
    raise SystemExit("Set JINA_API_KEY")

url = "https://segment.jina.ai/"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}",
}
text = "公司 2023 年 Q3 营收为 500 万美元。\n\n主要得益于 AI 产品的订阅增长。\n\n2024 年战略规划：全面投入 Agentic 架构研发，特别是 MAP-RAG 系统。"

data = {
    "content": text,
    "return_chunks": True,
}
response = requests.post(url, headers=headers, json=data)
print(response.json())

import requests

url = "https://segment.jina.ai/"
headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer jina_fc4562cfaf284cea94b01af1294d13c0LW8z0DPSfQJp8jkfVUeoAbVpIT40"
}
text = "公司 2023 年 Q3 营收为 500 万美元。\n\n主要得益于 AI 产品的订阅增长。\n\n2024 年战略规划：全面投入 Agentic 架构研发，特别是 MAP-RAG 系统。"

data = {
  "content": text,
  "return_chunks": True,
}
response = requests.post(url, headers=headers, json=data)
print(response.json())

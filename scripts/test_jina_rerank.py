import requests

url = "https://api.jina.ai/v1/rerank"
headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer jina_fc4562cfaf284cea94b01af1294d13c0LW8z0DPSfQJp8jkfVUeoAbVpIT40"
}
data = {
  "model": "jina-reranker-v2-base-multilingual",
  "query": "咱们公司未来打算搞什么技术架构？",
  "documents": [
      "开源社区贡献指南：鼓励员工参与。",
      "2024 年战略规划：全面投入 Agentic 架构研发，特别是 MAP-RAG 系统。"
  ],
  "top_n": 2
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(response.json())
except Exception as e:
    print(e)

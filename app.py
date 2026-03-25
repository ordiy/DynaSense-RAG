from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import uvicorn
from rag_core import process_document_task, run_chat_pipeline, run_evaluation

app = FastAPI(title="MAP-RAG MVP API")

# Mount static files for Web UI
import os
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Task tracking dictionary
tasks = {}

# Pydantic models for request bodies
class ChatRequest(BaseModel):
    query: str

class EvalRequest(BaseModel):
    query: str
    expected_substring: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()
    text_content = content.decode("utf-8")
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0}
    
    # Run async processing in background
    background_tasks.add_task(process_document_task, text_content, file.filename, tasks[task_id])
    
    return {"task_id": task_id, "message": "Document uploaded and processing started."}

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        result = run_chat_pipeline(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
async def evaluate(request: EvalRequest):
    try:
        result = run_evaluation(request.query, request.expected_substring)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

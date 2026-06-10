# FastAPI server
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import rag

# Create FastAPI app
app = FastAPI(
    title="DocuMind API",
    description="Self-healing RAG chatbot API",
    version="1.0.0"
)

# Allow frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request models
class SourceRequest(BaseModel):
    source_type: str
    source: str

class ChatRequest(BaseModel):
    question: str
    mode: str = "document"
    use_self_healing: bool = True

# Response models
class SourceResponse(BaseModel):
    message: str
    chunks: int

class ChatResponse(BaseModel):
    answer: str
    score: float
    label: str
    attempts: int

# Home endpoint
@app.get("/")
def home():
    return {"message": "DocuMind API is running"}

# Upload PDF file
@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF file"""
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = rag.process_source("pdf", file_path)
    os.remove(file_path)
    return SourceResponse(
        message=result["message"],
        chunks=result["chunks"]
    )

# Upload URL or YouTube or Text
@app.post("/upload/source")
async def upload_source(request: SourceRequest):
    """Upload URL, YouTube link or plain text"""
    result = rag.process_source(
        request.source_type,
        request.source
    )
    return SourceResponse(
        message=result["message"],
        chunks=result["chunks"]
    )

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with document or general chat"""
    
    # General chat mode
    if request.mode == "general":
        response = rag.llm.invoke(request.question).content
        return ChatResponse(
            answer=response,
            score=0.0,
            label="General chat",
            attempts=0
        )
    
    # Document mode - check if document loaded
    if rag.retriever is None:
        return {"error": "Please upload a document first!"}
    
    # Self healing or normal chat
    if request.use_self_healing:
        response, score, label, attempts = rag.self_healing_chat(
            request.question
        )
    else:
        response = rag.chat(request.question)
        score = 0.0
        label = "Normal chat"
        attempts = 1
    
    return ChatResponse(
        answer=response,
        score=score,
        label=label,
        attempts=attempts
    )
# Clear chat history
@app.post("/clear")
def clear():
    """Clear chat history"""
    rag.clear_history()
    return {"message": "Chat history cleared! ✅"}

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
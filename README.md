# DocuMind 🧠
### Self-healing RAG Chatbot

A production-ready RAG chatbot that can chat with any document, URL, YouTube video or plain text. Features automatic self-healing with chunk quality grading.

---

## Features

- 📄 Multi-source support — PDF, URL, YouTube, Text
- 🔄 Self-healing RAG — automatically retries if chunk quality is low
- 📊 Chunk grading — scores retrieved chunks out of 5.0
- 💬 Chat memory — remembers conversation history
- 🤖 General chat mode — chat without any document
- ⚡ Fast API backend — REST API for frontend integration

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline (LCEL) |
| Mistral AI | LLM + Embeddings |
| Pinecone | Vector database |
| FastAPI | REST API backend |
| Python | Core language |

---

## Project Structure
  documind
├── rag.py  
                     # RAG logic
├── rag.ipynb         # Development notebook
                       
├── main.py              # FastAPI server
                        
├── requirements.txt      # Dependencies
                           
└── .env             # API keys (not in repo)

---

## Setup

1. Clone the repo
```bash
git clone https://github.com/parth5980/self-healing-rag-chatbot.git
cd self-healing-rag-chatbot
```

2. Install dependencies
```bash
uv pip install -r requirements.txt
```

3. Add API keys in `.env`
MISTRAL_API_KEY=your_key
PINECONE_API_KEY=your_key

4. Run the server
```bash
python main.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /upload/file | Upload PDF |
| POST | /upload/source | Upload URL/YouTube/Text |
| POST | /chat | Chat with document |
| POST | /clear | Clear chat history |

---

## Self-healing Flow
User asks question
↓
Retrieve chunks from Pinecone
↓
Grade chunks (1.0 - 5.0)
↓
Score >= 3.0 → Generate answer ✅
Score < 3.0  → Retry with better query 🔄
↓
Max 3 attempts

---

## Author

**Parth** — Computer Engineering Student
Building AI/ML projects 🚀

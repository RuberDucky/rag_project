# RAG Chatbot with LangChain, Ollama, and pgvector

A powerful RAG (Retrieval Augmented Generation) chatbot for customer support using FastAPI, LangChain, Ollama, and PostgreSQL with pgvector.

## Features

- 📄 **Document Processing**: Upload and process PDF, DOCX, and TXT files
- 🧠 **Smart Embeddings**: Uses qwen3-embedding model via Ollama
- 🔍 **Vector Search**: HNSW indexing with pgvector for fast similarity search
- 💬 **Conversational Memory**: Maintains context across conversations
- 🤖 **LLM Integration**: Uses lfm2.5-thinking model for responses
- 🗄️ **Database**: Tortoise ORM with PostgreSQL

## Architecture

```
User → FastAPI → LangChain → qwen3-embedding (Ollama) → PostgreSQL + pgvector (HNSW) → Context → lfm2.5-thinking → Answer
```

## Prerequisites

- Python 3.14+
- PostgreSQL with pgvector extension
- Ollama server with models:
  - `lfm2.5-thinking`
  - `qwen3-embedding:latest`

## Setup

1. **Install dependencies**:
```bash
uv sync
```

2. **Configure environment** (`.env`):
```env
PORT=8000
DB_USERNAME=postgres
DB_PASSWORD=pass
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db

OLLAMA_BASE_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=qwen3-embedding:latest
CHAT_MODEL=lfm2.5-thinking

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Upload Configuration
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760

# Cleanup Configuration
DOCUMENT_RETENTION_DAYS=2
CLEANUP_INTERVAL_HOURS=24
```

3. **Run the application**:
```bash
uv run uvicorn main:app --reload
```

## API Endpoints

For frontend handoff with exact payload contracts, see [API_FRONTEND_GUIDE.md](API_FRONTEND_GUIDE.md).

### Documents
- `POST /documents/upload` - Upload and process documents
- `GET /documents/` - List all documents
- `GET /documents/{id}` - Get document details
- `DELETE /documents/{id}` - Delete document

### Chat
- `POST /chat/` - Send message and get AI response
- `POST /chat/stream` - Stream AI response via Server-Sent Events (SSE)
- `GET /chat/conversations` - List all conversations
- `GET /chat/conversations/{session_id}` - Get conversation details
- `DELETE /chat/conversations/{session_id}` - Delete conversation

### Health
- `GET /health` - Health check for database and Ollama

## Project Structure

```
rag_project/
├── main.py                 # FastAPI application
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── database.py        # Database initialization
│   ├── models.py          # Tortoise ORM models
│   ├── schemas.py         # Pydantic schemas
│   ├── ollama_client.py   # Ollama API client
│   ├── document_processor.py  # Document text extraction
│   ├── text_chunker.py    # Text splitting
│   ├── vector_store.py    # Vector operations with pgvector
│   ├── chat_engine.py     # RAG chat engine with memory
│   └── routes/
│       ├── documents.py   # Document endpoints
│       └── chat.py        # Chat endpoints
├── uploads/               # Uploaded files
└── pyproject.toml         # Dependencies

```

## Usage Examples

### Upload a Document
```bash
curl -X POST "http://localhost:8000/documents/upload?user_id=11111111-1111-1111-1111-111111111111" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

If `user_id` is not provided, the API creates one and returns it in the upload response.

Use `user_id` as a query parameter for document listing and deletion to keep data scoped by user.

### Start a Chat
```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your business hours?",
    "user_id": "11111111-1111-1111-1111-111111111111"
  }'
```

### Continue Conversation
```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Do you offer weekend support?",
    "user_id": "11111111-1111-1111-1111-111111111111",
    "session_id": "previous-session-id-uuid"
  }'
```

### Stream Chat Response (New Endpoint)

Use this endpoint for token-by-token rendering in the frontend without changing the existing `/chat/` API.

```bash
curl -N -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Extract all percentages from my resume",
    "user_id": "11111111-1111-1111-1111-111111111111"
  }'
```

The stream returns SSE events:

- `meta`: includes `user_id`, `session_id`, and retrieved `context`
- `token`: includes incremental `content` to append in UI
- `done`: includes final `response`, `user_id`, `session_id`, `context`, `timestamp`
- `error`: includes terminal error message

### Frontend Streaming Guide (Fetch + SSE Parsing)

```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Extract all percentages from my resume',
    user_id: '11111111-1111-1111-1111-111111111111',
    session_id: null,
  }),
});

if (!response.ok || !response.body) throw new Error('Stream request failed');

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
let assistantText = '';

while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const events = buffer.split('\n\n');
  buffer = events.pop() ?? '';

  for (const rawEvent of events) {
    const eventLine = rawEvent.split('\n').find((line) => line.startsWith('event: '));
    const dataLine = rawEvent.split('\n').find((line) => line.startsWith('data: '));
    if (!eventLine || !dataLine) continue;

    const eventType = eventLine.replace('event: ', '').trim();
    const payload = JSON.parse(dataLine.replace('data: ', ''));

    if (eventType === 'meta') {
      // Save session_id/user_id/context for future turns and citations
      console.log('meta', payload);
    }

    if (eventType === 'token') {
      assistantText += payload.content;
      // Render progressively in UI
      renderAssistantText(assistantText);
    }

    if (eventType === 'done') {
      // Final payload includes full response + metadata
      console.log('done', payload);
    }

    if (eventType === 'error') {
      console.error('stream error', payload.message);
    }
  }
}
```

## Features in Detail

### Document Processing
- Supports PDF, DOCX, DOC, and TXT files
- Automatic text extraction and chunking
- Configurable chunk size and overlap
- Metadata preservation

### Vector Search
- Uses pgvector HNSW indexing for fast similarity search
- Cosine similarity for relevance scoring
- Configurable top-k retrieval

### Conversational Memory
- Maintains conversation history
- Auto-generates conversation titles
- Context-aware responses
- Supports multiple concurrent conversations

### RAG Pipeline
1. User sends a message
2. System retrieves relevant document chunks using vector similarity
3. Conversation history is loaded
4. Context + history + user message sent to LLM
5. Response generated and stored

## Configuration

Edit `.env` file:
- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of chunks to retrieve (default: 5)
- `MAX_FILE_SIZE`: Max upload size in bytes (default: 10MB)
- `DOCUMENT_RETENTION_DAYS`: Delete uploaded docs older than this (default: 2)
- `CLEANUP_INTERVAL_HOURS`: Cleanup scheduler frequency (default: 24)

## Development

Access interactive API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT

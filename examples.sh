"""
Example usage of the RAG Chatbot API.
Demonstrates document upload and conversation flow.
"""


# ===================================================================
# 1. HEALTH CHECK
# ===================================================================

# Check if the API is running and all services are healthy
curl http://localhost:8000/health


# ===================================================================
# 2. UPLOAD DOCUMENTS
# ===================================================================

# Upload a PDF document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"

# Upload a DOCX document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.docx"

# Upload a TXT document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.txt"


# ===================================================================
# 3. LIST DOCUMENTS
# ===================================================================

# Get all uploaded documents
curl http://localhost:8000/documents/


# ===================================================================
# 4. START A CONVERSATION
# ===================================================================

# Send first message (creates new conversation)
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your business hours?"
  }'

# Response will include:
# - session_id: UUID for this conversation
# - response: AI generated answer
# - context: Relevant document chunks used


# ===================================================================
# 5. CONTINUE CONVERSATION
# ===================================================================

# Send follow-up message (use session_id from previous response)
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Do you offer weekend support?",
    "session_id": "your-session-id-here"
  }'


# ===================================================================
# 6. VIEW CONVERSATIONS
# ===================================================================

# List all conversations
curl http://localhost:8000/chat/conversations

# Get specific conversation details
curl http://localhost:8000/chat/conversations/{session_id}


# ===================================================================
# 7. DELETE CONVERSATION
# ===================================================================

# Delete a conversation and all its messages
curl -X DELETE "http://localhost:8000/chat/conversations/{session_id}"


# ===================================================================
# 8. DELETE DOCUMENT
# ===================================================================

# Delete a document and all its embeddings
curl -X DELETE "http://localhost:8000/documents/{document_id}"


# ===================================================================
# 9. EXAMPLE: COMPLETE WORKFLOW
# ===================================================================

# Step 1: Upload a document about your product/service
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@product_manual.pdf"

# Step 2: Start chatting
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I install the product?"
  }'

# Step 3: Ask follow-up (using session_id from step 2)
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the system requirements?",
    "session_id": "session-id-from-step-2"
  }'

# Step 4: Review the conversation
curl http://localhost:8000/chat/conversations


# ===================================================================
# 10. ACCESS API DOCUMENTATION
# ===================================================================

# Swagger UI (interactive docs)
# Open in browser: http://localhost:8000/docs

# ReDoc (alternative docs)
# Open in browser: http://localhost:8000/redoc

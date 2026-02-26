#!/usr/bin/env python3
"""
Test script for RAG Chatbot API
"""
import asyncio
import httpx
from pathlib import Path


BASE_URL = "http://localhost:8000"


async def test_health():
    """Test health check endpoint."""
    print("\n🏥 Testing Health Check...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200


async def test_upload_document(file_path: str):
    """Test document upload."""
    print(f"\n📄 Uploading document: {file_path}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = await client.post(
                f"{BASE_URL}/documents/upload",
                files=files
            )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.json() if response.status_code == 200 else None


async def test_list_documents():
    """Test listing documents."""
    print("\n📋 Listing all documents...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/documents/")
        print(f"Status: {response.status_code}")
        documents = response.json()
        print(f"Found {len(documents)} documents")
        for doc in documents:
            print(f"  - {doc['filename']} ({doc['file_type']}, {doc['chunk_count']} chunks)")
        return documents


async def test_chat(message: str, session_id: str = None):
    """Test chat endpoint."""
    print(f"\n💬 Sending message: {message}")
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {"message": message}
        current_user_id = getattr(test_chat, "user_id", None)
        if current_user_id:
            payload["user_id"] = current_user_id
        if session_id:
            payload["session_id"] = session_id
        
        response = await client.post(
            f"{BASE_URL}/chat/",
            json=payload
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Session ID: {data['session_id']}")
            print(f"User ID: {data['user_id']}")
            print(f"Response: {data['response']}")
            print(f"\nContext used ({len(data['context'])} sources):")
            for ctx in data['context']:
                print(f"  - {ctx['source']} (score: {ctx['score']:.3f})")
            return data
        else:
            print(f"Error: {response.text}")
            return None


async def test_conversations():
    """Test listing conversations."""
    print("\n💭 Listing all conversations...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/chat/conversations")
        print(f"Status: {response.status_code}")
        conversations = response.json()
        print(f"Found {len(conversations)} conversations")
        for conv in conversations:
            print(f"  - {conv['title']} ({conv['message_count']} messages)")
        return conversations


async def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("RAG CHATBOT API TEST SUITE")
    print("=" * 60)
    
    # Test health
    health_ok = await test_health()
    if not health_ok:
        print("\n❌ Health check failed! Make sure the server is running.")
        return
    
    # Test listing documents (initially)
    await test_list_documents()
    
    # Test chat without documents (should work but with limited context)
    print("\n" + "=" * 60)
    print("TESTING CHAT (without uploaded documents)")
    print("=" * 60)
    
    chat1 = await test_chat("Hello! How can you help me?")
    if chat1:
        test_chat.user_id = chat1['user_id']
        session_id = chat1['session_id']
        
        # Continue conversation
        await test_chat(
            "What information do you have available?",
            session_id=session_id
        )
    
    # Test listing conversations
    await test_conversations()
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)
    print("\nNOTE: To test with documents, upload a PDF/DOCX file:")
    print(f"  curl -X POST '{BASE_URL}/documents/upload' \\")
    print(f"    -F 'file=@/path/to/your/document.pdf'")
    print("\nThen chat again to see RAG in action!")


if __name__ == "__main__":
    test_chat.user_id = None
    asyncio.run(run_tests())

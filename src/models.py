"""Database models using Tortoise ORM."""
from tortoise import fields
from tortoise.models import Model
from datetime import datetime
from typing import Optional


class Document(Model):
    """Model for storing uploaded documents."""
    
    id = fields.IntField(pk=True)
    user_id = fields.UUIDField()
    filename = fields.CharField(max_length=255)
    original_filename = fields.CharField(max_length=255)
    file_type = fields.CharField(max_length=50)
    file_size = fields.IntField()
    upload_date = fields.DatetimeField(auto_now_add=True)
    processed = fields.BooleanField(default=False)
    chunk_count = fields.IntField(default=0)
    
    class Meta:
        table = "documents"
        
    def __str__(self):
        return f"Document({self.original_filename})"


class Conversation(Model):
    """Model for storing conversation sessions."""
    
    id = fields.IntField(pk=True)
    user_id = fields.UUIDField()
    session_id = fields.UUIDField(unique=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    title = fields.CharField(max_length=255, null=True)
    
    class Meta:
        table = "conversations"
        
    def __str__(self):
        return f"Conversation({self.session_id})"


class Message(Model):
    """Model for storing chat messages."""
    
    id = fields.IntField(pk=True)
    conversation = fields.ForeignKeyField(
        "models.Conversation",
        related_name="messages",
        on_delete=fields.CASCADE
    )
    role = fields.CharField(max_length=50)  # 'user' or 'assistant'
    content = fields.TextField()
    timestamp = fields.DatetimeField(auto_now_add=True)
    context_used = fields.JSONField(null=True)  # Store retrieved context
    
    class Meta:
        table = "messages"
        ordering = ["timestamp"]
        
    def __str__(self):
        return f"Message({self.role}: {self.content[:50]}...)"


class DocumentChunk(Model):
    """Model for storing document chunks with embeddings."""
    
    id = fields.IntField(pk=True)
    document = fields.ForeignKeyField(
        "models.Document",
        related_name="chunks",
        on_delete=fields.CASCADE
    )
    chunk_index = fields.IntField()
    content = fields.TextField()
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta:
        table = "document_chunks"
        unique_together = (("document", "chunk_index"),)
        
    def __str__(self):
        return f"DocumentChunk({self.document_id}, {self.chunk_index})"

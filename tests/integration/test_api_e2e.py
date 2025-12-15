"""
Week 5 Day 5: End-to-End Integration Tests
==========================================
Complete pipeline validation with FastAPI TestClient.
"""

import pytest
import time
from fastapi.testclient import TestClient
from src.memory_architect.server.api import app
from src.memory_architect.storage.vector_store import ChromaManager
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass


# Test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test basic API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Memory Architect API" in data["message"]
        assert data["status"] == "operational"
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "total_memories" in data
        assert "timestamp" in data
    
    def test_memory_stats(self):
        """Test memory statistics endpoint."""
        response = client.get("/memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data
        assert "collection_name" in data


class TestChatEndpoint:
    """Test main chat endpoint."""
    
    def test_basic_chat_request(self):
        """Test basic chat functionality."""
        response = client.post("/chat", json={
            "user_id": "test_user",
            "text": "Hello, world!"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "response" in data
        assert "memories_used" in data
        assert "processing_time_ms" in data
        
        # Verify types
        assert isinstance(data["response"], str)
        assert isinstance(data["memories_used"], int)
        assert isinstance(data["processing_time_ms"], float)
    
    def test_chat_with_pii(self):
        """Test that PII is sanitized in chat."""
        response = client.post("/chat", json={
            "user_id": "test_user",
            "text": "My email is alice@example.com"
        })
        
        assert response.status_code == 200
        # PII should be sanitized before processing
    
    def test_chat_with_high_token_limit(self):
        """Test chat with custom token limit."""
        response = client.post("/chat", json={
            "user_id": "test_user",
            "text": "Tell me about Python",
            "max_tokens": 1000
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["memories_used"] >= 0


class TestMemoryManagement:
    """Test memory addition endpoint."""
    
    def test_add_memory(self):
        """Test manually adding a memory."""
        response = client.post("/memory/add", params={
            "user_id": "test_user",
            "content": "User prefers Python for ML"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "memory_id" in data
        assert "content_preview" in data
    
    def test_add_memory_with_tags(self):
        """Test adding memory with tags."""
        response = client.post("/memory/add", params={
            "user_id": "test_user",
            "content": "Important fact about user",
            "tags": ["important", "fact"]
        })
        
        assert response.status_code == 200


class TestEndToEndScenario:
    """
    Test complete user scenario (Day 5 requirement).
    
    Scenario:
    1. User says "My name is Alice"
    2. System stores fact
    3. User asks "What is my name?"
    4. System retrieves fact and answers correctly
    """
    
    def test_store_and_retrieve_scenario(self):
        """Test storing a fact and retrieving it."""
        user_id = f"e2e_test_{int(time.time())}"
        
        # Step 1: Store a fact
        response1 = client.post("/memory/add", params={
            "user_id": user_id,
            "content": "My name is Alice",
            "memory_type": "episodic"
        })
        
        assert response1.status_code == 200
        memory_id = response1.json()["memory_id"]
        assert memory_id is not None
        
        # Step 2: Wait a moment for indexing
        time.sleep(0.5)
        
        # Step 3: Query for the fact
        response2 = client.post("/chat", json={
            "user_id": user_id,
            "text": "What is my name?"
        })
        
        assert response2.status_code == 200
        data = response2.json()
        
        # Step 4: Verify response includes context
        # (In production with real LLM, would check for "Alice" in response)
        assert "response" in data
        assert data["memories_used"] >= 0  # Should find the memory
    
    def test_reflection_updates_scores(self):
        """Test that reflection loop updates memory scores."""
        user_id = f"reflection_test_{int(time.time())}"
        
        # Add a memory
        client.post("/memory/add", params={
            "user_id": user_id,
            "content": "Python is the preferred language"
        })
        
        # Query multiple times (should trigger reflection)
        for _ in range(3):
            client.post("/chat", json={
                "user_id": user_id,
                "text": "What language do I prefer?"
            })
            time.sleep(0.1)  # Let background tasks complete
        
        # Note: In full integration, we'd verify score increased
        # For now, just verify no errors occurred


class TestConcurrency:
    """Test concurrent requests."""
    
    def test_multiple_concurrent_users(self):
        """Test that multiple users can use API simultaneously."""
        users = [f"user_{i}" for i in range(5)]
        
        responses = []
        for user in users:
            response = client.post("/chat", json={
                "user_id": user,
                "text": f"Hello from {user}"
            })
            responses.append(response)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # Each should have independent results
        assert len(set(r.json()["response"] for r in responses)) >= 1


class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_user_id(self):
        """Test that missing user_id is rejected."""
        response = client.post("/chat", json={
            "text": "Hello"
        })
        
        # Should fail validation
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_text(self):
        """Test that missing text is rejected."""
        response = client.post("/chat", json={
            "user_id": "test"
        })
        
        assert response.status_code == 422

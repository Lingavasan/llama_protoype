"""
Week 5 Day 1: FastAPI Interface
================================
REST API for Memory Architect system.
Exposes chat endpoint with background reflection tasks.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import time

# Import Memory Architect components
from src.memory_architect.storage.vector_store import ChromaManager
from src.memory_architect.policy.privacy import PrivacyGuard
from src.memory_architect.core.budgeter import greedy_knapsack_budgeting, estimate_token_budget
from src.memory_architect.core.prompt_assembler import assemble_context
from src.memory_architect.core.reflection import update_memory_scores
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass


# Mock embedding function for Python 3.13 compatibility
class MockEmbeddingFunction:
    """Mock embedding function for testing."""
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        return [[0.1, 0.2, 0.3, 0.4, 0.5] * 2 for _ in input]
    
    def embed_query(self, input: List[str]) -> List[List[float]]:
        """Generate mock query embeddings."""
        return self.__call__(input)
    
    def name(self) -> str:
        """Return the name of the embedding function."""
        return "mock-embedding-function"
    
    def is_legacy(self) -> bool:
        """Indicate if this is a legacy embedding function."""
        return True


# Initialize FastAPI app
app = FastAPI(
    title="Memory Architect API",
    description="Long-term memory system for LLMs with reflection and adaptation",
    version="1.0.0"
)

# Initialize components (use mock embedding for compatibility)
try:
    db = ChromaManager(persist_path="./data/.chroma")
except Exception:
    # Fallback to mock embedding for Python 3.13
    db = ChromaManager(persist_path="./data/.chroma", embedding_function=MockEmbeddingFunction())

privacy_guard = PrivacyGuard()


class QueryRequest(BaseModel):
    """Chat query request."""
    user_id: str = Field(..., description="User ID for memory isolation")
    text: str = Field(..., description="User's message text")
    role: str = Field(default="user", description="User role (user|admin)")
    max_tokens: int = Field(default=2000, description="Maximum tokens for context")


class QueryResponse(BaseModel):
    """Chat query response."""
    response: str = Field(..., description="LLM generated response")
    memories_used: int = Field(..., description="Number of memories in context")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class MemoryStats(BaseModel):
    """Memory statistics."""
    total_memories: int
    collection_name: str


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Memory Architect API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=dict, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and basic metrics.
    """
    try:
        stats = db.get_collection_stats()
        return {
            "status": "healthy",
            "total_memories": stats.get('total_memories', 0),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/chat", response_model=QueryResponse, tags=["Chat"])
async def chat_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Main chat endpoint with memory integration.
    
    Pipeline:
    1. PII sanitization
    2. Memory retrieval + ranking
    3. Budget selection
    4. Prompt assembly
    5. LLM generation (mocked for now)
    6. Background reflection task
    
    Args:
        request: Chat query with user_id and text
        background_tasks: FastAPI background task manager
    
    Returns:
        Response with LLM output and metadata
    """
    start_time = time.time()
    
    try:
        # 1. PII Sanitization
        clean_text = privacy_guard.sanitize(
            request.text,
            role=request.role
        )
        
        # 2. Memory Retrieval
        # First, retrieve candidates with metadata filtering
        raw_results = db.retrieve_candidates(
            query_text=clean_text,
            user_id=request.user_id,
            k=20  # Retrieve more candidates for ranking
        )
        
        # 3. Hybrid Ranking
        ranked_results = db.rank_results(raw_results)
        
        # Convert to budget format
        candidates = [
            {
                'content': metadata.get('content', ''),
                'score': score
            }
            for memory_id, score, metadata in ranked_results
        ]
        
        # 4. Budget Selection
        budget = min(request.max_tokens, estimate_token_budget(4096))
        selected_contents = greedy_knapsack_budgeting(candidates, budget)
        
        # 5. Prompt Assembly
        prompt = assemble_context(selected_contents, clean_text)
        
        # 6. LLM Generation (MOCKED - integrate your LLM here)
        # In production, replace with actual LLM call:
        # response_text = llm_client.generate(prompt)
        response_text = f"[MOCK LLM RESPONSE] I found {len(selected_contents)} relevant memories about: {clean_text[:50]}..."
        
        # 7. Background Reflection Task
        # This runs asynchronously after response is sent
        if ranked_results:
            # Convert ranked results to MemoryChunks for reflection
            retrieved_chunks = []
            for memory_id, score, metadata in ranked_results[:10]:  # Top 10
                chunk = MemoryChunk(
                    id=memory_id,
                    content=metadata.get('content', ''),
                    type=MemoryType(metadata.get('type', 'episodic')),
                    policy=PolicyClass(metadata.get('policy', 'ephemeral')),
                    source_session_id=metadata.get('source_session_id', ''),
                    user_id=metadata['user_id'],
                    tags=metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                    reflection_score=metadata.get('reflection_score', 50.0),
                    created_at=metadata.get('created_at', time.time()),
                    last_accessed=metadata.get('last_accessed', time.time())
                )
                retrieved_chunks.append(chunk)
            
            background_tasks.add_task(
                update_memory_scores,
                retrieved_chunks,
                response_text,
                db
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return QueryResponse(
            response=response_text,
            memories_used=len(selected_contents),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@app.post("/memory/add", tags=["Memory"])
async def add_memory(
    user_id: str,
    content: str,
    memory_type: str = "episodic",
    tags: Optional[List[str]] = None
):
    """
    Manually add a memory to the system.
    
    Useful for:
    - Pre-loading user profiles
    - CLI memory injection
    - Testing
    
    Args:
        user_id: User ID
        content: Memory content
        memory_type: Type (episodic|semantic|procedural)
        tags: Optional tags
    
    Returns:
        Memory ID and status
    """
    try:
        # Sanitize content
        clean_content = privacy_guard.sanitize(content, role="admin")
        
        # Create memory chunk
        chunk = MemoryChunk(
            content=clean_content,
            type=MemoryType(memory_type),
            policy=PolicyClass.EPHEMERAL,
            source_session_id="api_manual",
            user_id=user_id,
            tags=tags or [],
            reflection_score=50.0,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        # Store
        db.add_memory(chunk)
        
        return {
            "status": "success",
            "memory_id": chunk.id,
            "content_preview": clean_content[:100]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add memory: {str(e)}"
        )


@app.get("/memory/stats", response_model=MemoryStats, tags=["Memory"])
async def get_memory_stats():
    """
    Get memory collection statistics.
    
    Returns counts and metadata about stored memories.
    """
    try:
        stats = db.get_collection_stats()
        return MemoryStats(
            total_memories=stats.get('total_memories', 0),
            collection_name=stats.get('collection_name', 'unknown')
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

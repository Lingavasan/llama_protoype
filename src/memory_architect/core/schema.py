from enum import Enum
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class MemoryType(str, Enum):
    EPISODIC = "episodic"   # Specific events, highly time-dependent
    SEMANTIC = "semantic"   # General facts, time-independent
    PROCEDURAL = "procedural" # How-to knowledge

class PolicyClass(str, Enum):
    CANONICAL = "canonical" # Never expires (e.g., core personality)
    EPHEMERAL = "ephemeral" # Expires quickly (e.g., chat chitchat)
    SENSITIVE = "sensitive" # Requires strict redaction and limited TTL

class MemoryChunk(BaseModel):
    """
    The atomic unit of memory stored in the Vector DB.
    Encapsulates content, embedding, and governance metadata.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    vector: Optional[List[float]] = None # Computed embedding
    
    # Governance Metadata
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    last_accessed: float = Field(default_factory=lambda: datetime.now().timestamp())
    access_count: int = 0
    
    # Policy Drivers
    type: MemoryType
    policy: PolicyClass
    reflection_score: float = 50.0  # Starts neutral (0-100)
    ttl_seconds: Optional[int] = None
    
    # Context Tracking
    source_session_id: str
    user_id: str
    tags: List[str] = Field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Checks if the memory has exceeded its Time-To-Live."""
        if self.policy == PolicyClass.CANONICAL:
            return False
        if self.ttl_seconds is None:
            return False
        now = datetime.now().timestamp()
        return (now - self.created_at) > self.ttl_seconds

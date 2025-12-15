import time
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass

def test_schema():
    print("Testing MemoryChunk schema...")
    
    # Test creation
    mem = MemoryChunk(
        content="The sky is blue.",
        type=MemoryType.SEMANTIC,
        policy=PolicyClass.CANONICAL,
        source_session_id="sess-1",
        user_id="user-1"
    )
    print("MemoryChunk created successfully.")
    assert mem.id is not None
    assert mem.created_at > 0
    assert mem.tags == []
    
    # Test Expiration - Canonical (should never expire)
    mem.ttl_seconds = 1
    time.sleep(1.1)
    assert not mem.is_expired()
    print("Canonical memory did not expire (correct).")
    
    # Test Expiration - Ephemeral
    mem_eph = MemoryChunk(
        content="Hello there",
        type=MemoryType.EPISODIC,
        policy=PolicyClass.EPHEMERAL,
        ttl_seconds=1,
        source_session_id="sess-1",
        user_id="user-1"
    )
    assert not mem_eph.is_expired()
    time.sleep(1.1)
    assert mem_eph.is_expired()
    print("Ephemeral memory expired (correct).")
    
    print("All assertions passed.")

if __name__ == "__main__":
    test_schema()

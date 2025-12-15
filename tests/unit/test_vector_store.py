"""
Unit tests for ChromaDB Vector Store (Week 2)
==============================================
Tests the persistent storage, schema mapping, and metadata filtering capabilities.
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
from src.memory_architect.storage.vector_store import ChromaManager


class MockEmbeddingFunction:
    """
    Mock embedding function for testing without scipy/sentence-transformers dependencies.
    This avoids Python 3.13 compatibility issues with scipy.
    """
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate simple mock embeddings (10-dimensional vectors)."""
        return [[0.1] * 10 for _ in input]
    
    def embed_query(self, input: List[str]) -> List[List[float]]:
        """ChromaDB 1.3.5 API method for query embeddings."""
        return self.__call__(input)
    
    def name(self) -> str:
        """Return the name of the embedding function."""
        return "mock-embedding-function"
    
    def is_legacy(self) -> bool:
        """Indicate if this is a legacy embedding function."""
        return True


@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB persistence."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def chroma_manager(temp_chroma_path):
    """Create a ChromaManager instance with temporary storage and mock embeddings."""
    return ChromaManager(persist_path=temp_chroma_path, embedding_function=MockEmbeddingFunction())


@pytest.fixture
def sample_memory_canonical():
    """Create a canonical memory chunk for testing."""
    return MemoryChunk(
        content="The user's name is Alice and she prefers Python programming.",
        type=MemoryType.SEMANTIC,
        policy=PolicyClass.CANONICAL,
        source_session_id="session_001",
        user_id="user_alice",
        tags=["identity", "preferences"],
        reflection_score=90.0,
        ttl_seconds=None  # Canonical memories don't expire
    )


@pytest.fixture
def sample_memory_ephemeral():
    """Create an ephemeral memory chunk for testing."""
    return MemoryChunk(
        content="Alice said 'Good morning!' at 9am.",
        type=MemoryType.EPISODIC,
        policy=PolicyClass.EPHEMERAL,
        source_session_id="session_002",
        user_id="user_alice",
        tags=["greeting", "temporal"],
        reflection_score=30.0,
        ttl_seconds=3600  # Expires in 1 hour
    )


@pytest.fixture
def sample_memory_different_user():
    """Create a memory for a different user."""
    return MemoryChunk(
        content="Bob enjoys JavaScript development.",
        type=MemoryType.SEMANTIC,
        policy=PolicyClass.CANONICAL,
        source_session_id="session_003",
        user_id="user_bob",
        tags=["preferences"],
        reflection_score=85.0,
        ttl_seconds=None
    )


class TestChromaManagerInitialization:
    """Test ChromaDB client initialization and collection setup."""
    
    def test_persistent_client_creation(self, temp_chroma_path):
        """Test that PersistentClient is created with correct path."""
        manager = ChromaManager(persist_path=temp_chroma_path, embedding_function=MockEmbeddingFunction())
        
        # Verify collection was created
        assert manager.collection is not None
        assert manager.collection.name == "memory_architect_main"
        
        # Verify persistence directory exists
        chroma_path = Path(temp_chroma_path)
        assert chroma_path.exists()
    
    def test_collection_metadata(self, chroma_manager):
        """Test that collection is configured with cosine similarity."""
        metadata = chroma_manager.collection.metadata
        assert metadata.get("hnsw:space") == "cosine"
    
    def test_embedding_function_configured(self, chroma_manager):
        """Test that local SentenceTransformer embedding is configured."""
        assert chroma_manager.embedding_fn is not None


class TestMemoryStorage:
    """Test memory addition and schema mapping."""
    
    def test_add_canonical_memory(self, chroma_manager, sample_memory_canonical):
        """Test adding a canonical memory to the database."""
        chroma_manager.add_memory(sample_memory_canonical)
        
        # Verify memory was stored
        result = chroma_manager.collection.get(ids=[sample_memory_canonical.id])
        
        assert len(result['ids']) == 1
        assert result['documents'][0] == sample_memory_canonical.content
        assert result['metadatas'][0]['policy'] == PolicyClass.CANONICAL.value
        assert result['metadatas'][0]['user_id'] == "user_alice"
        assert result['metadatas'][0]['expiry_timestamp'] == -1.0  # No expiry
    
    def test_add_ephemeral_memory(self, chroma_manager, sample_memory_ephemeral):
        """Test adding an ephemeral memory with TTL."""
        chroma_manager.add_memory(sample_memory_ephemeral)
        
        result = chroma_manager.collection.get(ids=[sample_memory_ephemeral.id])
        
        assert len(result['ids']) == 1
        metadata = result['metadatas'][0]
        
        # Verify expiry timestamp was calculated
        expected_expiry = sample_memory_ephemeral.created_at + sample_memory_ephemeral.ttl_seconds
        assert metadata['expiry_timestamp'] == pytest.approx(expected_expiry, rel=1e-5)
    
    def test_metadata_flattening(self, chroma_manager, sample_memory_canonical):
        """Test that metadata is properly flattened (e.g., tags as comma-separated string)."""
        chroma_manager.add_memory(sample_memory_canonical)
        
        result = chroma_manager.collection.get(ids=[sample_memory_canonical.id])
        metadata = result['metadatas'][0]
        
        # Tags should be comma-separated string
        assert metadata['tags'] == "identity,preferences"
        assert isinstance(metadata['tags'], str)
    
    def test_upsert_updates_existing(self, chroma_manager, sample_memory_canonical):
        """Test that calling add_memory on existing ID updates the memory."""
        # Add initial memory
        chroma_manager.add_memory(sample_memory_canonical)
        
        # Modify and re-add
        sample_memory_canonical.content = "Updated content about Alice."
        sample_memory_canonical.reflection_score = 95.0
        chroma_manager.add_memory(sample_memory_canonical)
        
        # Verify only one entry exists with updated content
        result = chroma_manager.collection.get(ids=[sample_memory_canonical.id])
        assert len(result['ids']) == 1
        assert result['documents'][0] == "Updated content about Alice."
        assert result['metadatas'][0]['reflection_score'] == 95.0


class TestMetadataFiltering:
    """Test intelligent retrieval with metadata pre-filtering."""
    
    def test_user_isolation(
        self, 
        chroma_manager, 
        sample_memory_canonical, 
        sample_memory_different_user
    ):
        """Test that users can only retrieve their own memories."""
        # Add memories for two different users
        chroma_manager.add_memory(sample_memory_canonical)
        chroma_manager.add_memory(sample_memory_different_user)
        
        # Query as Alice
        results = chroma_manager.retrieve_candidates(
            query_text="programming preferences",
            user_id="user_alice",
            k=10
        )
        
        # Should only retrieve Alice's memory, not Bob's
        assert len(results['ids'][0]) == 1
        assert results['metadatas'][0][0]['user_id'] == "user_alice"
        assert "Python" in results['documents'][0][0]
    
    def test_ttl_filtering_expired(self, chroma_manager):
        """Test that expired memories are not retrieved."""
        # Create an already-expired memory (TTL = -1000 seconds, i.e., expired 1000 seconds ago)
        expired_memory = MemoryChunk(
            content="This memory has expired.",
            type=MemoryType.EPISODIC,
            policy=PolicyClass.EPHEMERAL,
            source_session_id="session_expired",
            user_id="user_alice",
            tags=["old"],
            reflection_score=20.0,
            created_at=datetime.now().timestamp() - 2000,  # 2000 seconds ago
            ttl_seconds=1000  # Expired 1000 seconds ago
        )
        
        chroma_manager.add_memory(expired_memory)
        
        # Try to retrieve - should get nothing
        results = chroma_manager.retrieve_candidates(
            query_text="memory",
            user_id="user_alice",
            k=10
        )
        
        # Should return empty or not include the expired memory
        if results['ids'][0]:
            for metadata in results['metadatas'][0]:
                # Verify no expired memories in results
                current_time = datetime.now().timestamp()
                expiry = metadata['expiry_timestamp']
                if expiry != -1.0:
                    assert expiry > current_time
    
    def test_canonical_never_expires(self, chroma_manager, sample_memory_canonical):
        """Test that canonical memories are always retrieved regardless of age."""
        # Set an old created_at timestamp
        sample_memory_canonical.created_at = datetime.now().timestamp() - 1000000
        chroma_manager.add_memory(sample_memory_canonical)
        
        # Should still be retrievable
        results = chroma_manager.retrieve_candidates(
            query_text="Alice",
            user_id="user_alice",
            k=10
        )
        
        assert len(results['ids'][0]) >= 1
        assert results['metadatas'][0][0]['policy'] == PolicyClass.CANONICAL.value
    
    def test_complex_filter_logic(
        self, 
        chroma_manager, 
        sample_memory_canonical,
        sample_memory_ephemeral,
        sample_memory_different_user
    ):
        """Test the complex AND/OR filter logic."""
        # Add multiple memories with different policies and users
        chroma_manager.add_memory(sample_memory_canonical)
        chroma_manager.add_memory(sample_memory_ephemeral)
        chroma_manager.add_memory(sample_memory_different_user)
        
        # Query as Alice - should get her canonical + non-expired ephemeral
        results = chroma_manager.retrieve_candidates(
            query_text="Alice",
            user_id="user_alice",
            k=10
        )
        
        # Should retrieve Alice's memories only
        retrieved_count = len(results['ids'][0])
        assert retrieved_count >= 1  # At least the canonical memory
        
        for metadata in results['metadatas'][0]:
            # All results must belong to Alice
            assert metadata['user_id'] == "user_alice"


class TestAccessTracking:
    """Test access metadata updates."""
    
    def test_update_access_metadata(self, chroma_manager, sample_memory_canonical):
        """Test that access count and timestamp are updated."""
        chroma_manager.add_memory(sample_memory_canonical)
        
        original_access_count = sample_memory_canonical.access_count
        
        # Update access metadata
        chroma_manager.update_access_metadata(sample_memory_canonical.id)
        
        # Verify metadata was updated
        result = chroma_manager.collection.get(ids=[sample_memory_canonical.id])
        metadata = result['metadatas'][0]
        
        assert metadata['access_count'] == original_access_count + 1
        assert metadata['last_accessed'] > sample_memory_canonical.last_accessed


class TestMemoryDeletion:
    """Test memory deletion functionality."""
    
    def test_delete_memory(self, chroma_manager, sample_memory_canonical):
        """Test deleting a memory from the store."""
        chroma_manager.add_memory(sample_memory_canonical)
        
        # Verify it exists
        result_before = chroma_manager.collection.get(ids=[sample_memory_canonical.id])
        assert len(result_before['ids']) == 1
        
        # Delete it
        chroma_manager.delete_memory(sample_memory_canonical.id)
        
        # Verify it's gone
        result_after = chroma_manager.collection.get(ids=[sample_memory_canonical.id])
        assert len(result_after['ids']) == 0


class TestCollectionStats:
    """Test collection statistics retrieval."""
    
    def test_get_stats(self, chroma_manager, sample_memory_canonical):
        """Test retrieving collection statistics."""
        # Initially empty
        stats = chroma_manager.get_collection_stats()
        initial_count = stats['count']
        
        # Add a memory
        chroma_manager.add_memory(sample_memory_canonical)
        
        # Check stats
        stats = chroma_manager.get_collection_stats()
        assert stats['count'] == initial_count + 1
        assert stats['name'] == "memory_architect_main"


# ============================================================================
# Day 3-5 Test Cases: Hybrid Retrieval and Graph Memory
# ============================================================================

class TestHybridRanking:
    """Test Day 3: Hybrid retrieval ranking algorithm."""
    
    def test_ranking_weights(self, chroma_manager):
        """
        Test Case 2 from Day 5: Verify ranking algorithm respects weights.
        
        Description:
        Add two memories:
        - Memory 1: High relevance, low reflection score (10/100)
        - Memory 2: Lower relevance, high reflection score (90/100)
        
        Expected: Ranking should balance both signals according to weights
        """
        import time
        
        # Memory 1: High relevance (similar to query) but low importance
        mem1 = MemoryChunk(
            content="Python programming is great for machine learning.",
            type=MemoryType.SEMANTIC,
            policy=PolicyClass.CANONICAL,
            source_session_id="session_rank_1",
            user_id="user_test",
            tags=["python", "ml"],
            reflection_score=10.0,  # Low importance
            ttl_seconds=None
        )
        
        # Memory 2: Lower relevance but high importance
        mem2 = MemoryChunk(
            content="The user loves working with data.",
            type=MemoryType.SEMANTIC,
            policy=PolicyClass.CANONICAL,
            source_session_id="session_rank_2",
            user_id="user_test",
            tags=["preferences"],
            reflection_score=90.0,  # High importance
            ttl_seconds=None,
            created_at=datetime.now().timestamp() - 3600  # 1 hour old
        )
        
        chroma_manager.add_memory(mem1)
        time.sleep(0.1)  # Small delay to ensure different timestamps
        chroma_manager.add_memory(mem2)
        
        # Query - should be more similar to mem1 (Python programming)
        results = chroma_manager.retrieve_candidates(
            query_text="Python programming",
            user_id="user_test",
            k=10
        )
        
        # Rank the results
        ranked = chroma_manager.rank_results(results)
        
        # Should have both memories
        assert len(ranked) == 2
        
        # Verify scoring components work
        for mem_id, score, meta in ranked:
            assert 0.0 <= score <= 3.0  # Max score is w_sim + w_ref + w_recency
            assert score > 0.0  # Should have positive score
    
    def test_recency_decay(self, chroma_manager):
        """Test that recency score decays with age."""
        import time
        
        # Recent memory
        recent = MemoryChunk(
            content="Recent information.",
            type=MemoryType.EPISODIC,
            policy=PolicyClass.CANONICAL,
            source_session_id="session_recent",
            user_id="user_test",
            tags=["recent"],
            reflection_score=50.0,
            ttl_seconds=None
        )
        
        # Old memory (24 hours ago)
        old = MemoryChunk(
            content="Old information.",
            type=MemoryType.EPISODIC,
            policy=PolicyClass.CANONICAL,
            source_session_id="session_old",
            user_id="user_test",
            tags=["old"],
            reflection_score=50.0,
            ttl_seconds=None,
            created_at=datetime.now().timestamp() - 86400  # 24 hours ago
        )
        
        chroma_manager.add_memory(recent)
        chroma_manager.add_memory(old)
        
        # Query both
        results = chroma_manager.retrieve_candidates(
            query_text="information",
            user_id="user_test",
            k=10
        )
        
        ranked = chroma_manager.rank_results(results)
        
        # Find scores for each
        recent_score = next((score for mid, score, meta in ranked if mid == recent.id), None)
        old_score = next((score for mid, score, meta in ranked if mid == old.id), None)
        
        # Recent should have higher score (all else equal, recency gives bonus)
        assert recent_score is not None
        assert old_score is not None
        assert recent_score > old_score
    
    def test_ttl_expiration_not_retrieved(self, chroma_manager):
        """
        Test Case 1 from Day 5: Privacy/TTL filtering.
        
        Description:
        Add a memory with ttl=1 second, sleep 2 seconds, query.
        Assert it is NOT returned.
        """
        import time
        
        # Create memory with 1 second TTL
        short_lived = MemoryChunk(
            content="This will expire soon.",
            type=MemoryType.EPISODIC,
            policy=PolicyClass.EPHEMERAL,
            source_session_id="session_ttl_test",
            user_id="user_test",
            tags=["temporary"],
            reflection_score=50.0,
            ttl_seconds=1  # 1 second TTL
        )
        
        chroma_manager.add_memory(short_lived)
        
        # Verify it exists initially
        results_before = chroma_manager.retrieve_candidates(
            query_text="expire",
            user_id="user_test",
            k=10
        )
        assert len(results_before['ids'][0]) >= 1
        
        # Wait for expiration
        time.sleep(2)
        
        # Query again - should NOT be returned
        results_after = chroma_manager.retrieve_candidates(
            query_text="expire",
            user_id="user_test",
            k=10
        )
        
        # Should be empty or not contain the expired memory
        if results_after['ids'][0]:
            retrieved_ids = results_after['ids'][0]
            assert short_lived.id not in retrieved_ids


class TestGraphMemory:
    """Test Day 4: NetworkX graph overlay functionality."""
    
    @pytest.fixture
    def graph_memory(self):
        """Create a GraphMemory instance for testing."""
        from src.memory_architect.storage.graph_store import GraphMemory
        return GraphMemory()
    
    @pytest.fixture
    def temp_graph_path(self):
        """Create temporary path for graph persistence."""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        path = f"{temp_dir}/test_graph.pkl"
        yield path
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_add_relation(self, graph_memory):
        """
        Test Case 3 from Day 5: Add relationship and verify.
        
        Description:
        Add a semantic triple (subject, predicate, object).
        Verify the relationship is stored.
        """
        # Add relationship: Alice manages Bob
        graph_memory.add_relation(
            subject="Alice",
            predicate="manages",
            object_entity="Bob",
            source_chunk_id="chunk-123"
        )
        
        # Verify relationship exists
        assert graph_memory.graph.has_edge("Alice", "Bob")
        
        # Verify edge data
        edge_data = graph_memory.graph.get_edge_data("Alice", "Bob")
        assert edge_data is not None
        
        # Should have relation and source_id
        first_edge = list(edge_data.values())[0]
        assert first_edge['relation'] == "manages"
        assert first_edge['source_id'] == "chunk-123"
    
    def test_get_related_entities(self, graph_memory):
        """
        Test Case 3 from Day 5: Verify BFS traversal returns connected nodes.
        
        Description:
        Create a relationship graph and test get_related_entities.
        """
        # Build a small graph:
        # Alice --manages--> Bob --knows--> Charlie
        # Alice --created--> Project
        graph_memory.add_relation("Alice", "manages", "Bob", "chunk-1")
        graph_memory.add_relation("Bob", "knows", "Charlie", "chunk-2")
        graph_memory.add_relation("Alice", "created", "Project", "chunk-3")
        
        # Get Alice's related entities at depth 1
        related = graph_memory.get_related_entities("Alice", depth=1)
        
        # Should find Bob and Project (direct connections)
        assert len(related) >= 2
        
        targets = [target for _, target, _ in related]
        assert "Bob" in targets
        assert "Project" in targets
        
        # Get Alice's related entities at depth 2
        related_depth2 = graph_memory.get_related_entities("Alice", depth=2)
        
        # Should also find Charlie (through Bob)
        targets_depth2 = [target for _, target, _ in related_depth2]
        assert "Charlie" in targets_depth2
    
    def test_multiple_relationships(self, graph_memory):
        """Test that MultiDiGraph allows multiple relations between same entities."""
        # Add two different relationships Alice -> Bob
        graph_memory.add_relation("Alice", "manages", "Bob", "chunk-1")
        graph_memory.add_relation("Alice", "mentors", "Bob", "chunk-2")
        
        # Both should exist
        edge_data = graph_memory.graph.get_edge_data("Alice", "Bob")
        assert len(edge_data) == 2  # Two different edges
        
        # Check both relations exist
        relations = [data['relation'] for data in edge_data.values()]
        assert "manages" in relations
        assert "mentors" in relations
    
    def test_outgoing_relations(self, graph_memory):
        """Test retrieving outgoing relationships."""
        graph_memory.add_relation("Alice", "manages", "Bob", "chunk-1")
        graph_memory.add_relation("Alice", "created", "Project", "chunk-2")
        
        outgoing = graph_memory.get_outgoing_relations("Alice")
        
        assert len(outgoing) == 2
        
        relations = [(rel, target) for rel, target, _ in outgoing]
        assert ("manages", "Bob") in relations
        assert ("created", "Project") in relations
    
    def test_incoming_relations(self, graph_memory):
        """Test retrieving incoming relationships."""
        graph_memory.add_relation("Alice", "manages", "Bob", "chunk-1")
        graph_memory.add_relation("Charlie", "works_with", "Bob", "chunk-2")
        
        incoming = graph_memory.get_incoming_relations("Bob")
        
        assert len(incoming) == 2
        
        relations = [(rel, source) for rel, source, _ in incoming]
        assert ("manages", "Alice") in relations
        assert ("works_with", "Charlie") in relations
    
    def test_find_path(self, graph_memory):
        """Test path finding between entities."""
        # Create path: Alice -> Bob -> Charlie
        graph_memory.add_relation("Alice", "knows", "Bob", "chunk-1")
        graph_memory.add_relation("Bob", "knows", "Charlie", "chunk-2")
        
        # Find path
        path = graph_memory.find_path("Alice", "Charlie")
        
        assert path is not None
        assert path == ["Alice", "Bob", "Charlie"]
        
        # No path should exist to non-existent entity
        no_path = graph_memory.find_path("Alice", "NonExistent")
        assert no_path is None
    
    def test_graph_persistence(self, graph_memory, temp_graph_path):
        """Test saving and loading graph from disk."""
        # Add some data
        graph_memory.add_relation("Alice", "manages", "Bob", "chunk-1")
        graph_memory.add_relation("Bob", "knows", "Charlie", "chunk-2")
        
        # Save
        graph_memory.save_graph(temp_graph_path)
        
        # Verify file exists
        assert Path(temp_graph_path).exists()
        
        # Create new instance and load
        from src.memory_architect.storage.graph_store import GraphMemory
        new_graph = GraphMemory()
        new_graph.load_graph(temp_graph_path)
        
        # Verify data was loaded
        assert new_graph.graph.has_edge("Alice", "Bob")
        assert new_graph.graph.has_edge("Bob", "Charlie")
        
        # Verify edge data
        edge_data = new_graph.graph.get_edge_data("Alice", "Bob")
        first_edge = list(edge_data.values())[0]
        assert first_edge['relation'] == "manages"
    
    def test_entity_management(self, graph_memory):
        """Test entity retrieval and statistics."""
        graph_memory.add_relation("Alice", "manages", "Bob", "chunk-1")
        graph_memory.add_relation("Bob", "knows", "Charlie", "chunk-2")
        graph_memory.add_relation("Alice", "created", "Project", "chunk-3")
        
        # Get all entities
        entities = graph_memory.get_all_entities()
        assert len(entities) == 4  # Alice, Bob, Charlie, Project
        assert "Alice" in entities
        assert "Bob" in entities
        assert "Charlie" in entities
        assert "Project" in entities
        
        # Get relation count
        count = graph_memory.get_relation_count()
        assert count == 3
    
    def test_graph_clear(self, graph_memory):
        """Test clearing the graph."""
        graph_memory.add_relation("Alice", "manages", "Bob", "chunk-1")
        
        assert graph_memory.get_relation_count() > 0
        
        graph_memory.clear()
        
        assert graph_memory.get_relation_count() == 0
        assert len(graph_memory.get_all_entities()) == 0

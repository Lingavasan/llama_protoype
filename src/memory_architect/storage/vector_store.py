"""
ChromaDB Vector Store (The Hippocampus)
=======================================
This is the long-term memory center.
It stores thoughts as "vectors" (numbers) so the AI can recall them later.
"""

import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.memory_architect.core.schema import MemoryChunk, PolicyClass


class ChromaManager:
    """
    The Librarian for our long-term memory.
    
    It uses ChromaDB to:
    1. SAVE facts forever (in the ./data/.chroma folder).
    2. FIND relevant facts when you ask a question.
    3. FILTER out stuff that isn't yours (Privacy).
    """
    
    def __init__(self, persist_path: str = "./data/.chroma", embedding_function=None):
        """
        Open the library.
        If the folder doesn't exist, we create it.
        """
        # Connect to the database on disk
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # We need a way to turn text into numbers (Embeddings).
        # We use a simple, fast model that runs right here on your laptop.
        # No API calls = No data leaving your machine.
        if embedding_function is None:
            # Default to a standard, reliable model (all-MiniLM-L6-v2)
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_fn = embedding_function
        
        # Open the specific "book" (collection) where we write our notes
        self.collection = self.client.get_or_create_collection(
            name="memory_architect_main",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}  # "Cosine" helps us find similar meanings
        )
    
    def add_memory(self, chunk: MemoryChunk) -> None:
        """
        Write a new memory into the book.
        
        We also stick a "Use By" label on it if it's temporary (TTL).
        """
        # If it has an expiration date, calculate when it should die.
        expiry_ts = -1.0
        if chunk.ttl_seconds is not None:
            expiry_ts = chunk.created_at + chunk.ttl_seconds
        
        # Pack up all the extra info (tags, score, user ID)
        metadata = {
            "type": chunk.type.value,
            "policy": chunk.policy.value,
            "reflection_score": chunk.reflection_score,
            "created_at": chunk.created_at,
            "last_accessed": chunk.last_accessed,
            "access_count": chunk.access_count,
            "user_id": chunk.user_id,
            "source_session_id": chunk.source_session_id,
            "expiry_timestamp": expiry_ts,
            "tags": ",".join(chunk.tags)  # Chroma wants strings, not lists, sorry.
        }
        
        # Save it! (Upsert means "Insert or Update if exists")
        self.collection.upsert(
            ids=[chunk.id],
            documents=[chunk.content],
            metadatas=[metadata]
        )
    
    def retrieve_candidates(
        self, 
        query_text: str, 
        user_id: str, 
        k: int = 10,
        limit_to_user: bool = True
    ) -> Dict[str, Any]:
        """
        The Search Party. Find memories related to `query_text`.
        
        Crucially, this acts as a FIREWALL:
        - If `limit_to_user` is True, it ONLY looks at YOUR memories.
        - It ignores anything that has Expired.
        """
        current_time = datetime.now().timestamp()
        
        # Build the security filter
        if limit_to_user:
            # STRICT MODE: Only My Data + Not Expired
            where_filter = {
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {
                        "$or": [
                            {"expiry_timestamp": {"$eq": -1.0}},      # Permanent
                            {"expiry_timestamp": {"$gt": current_time}} # Not dead yet
                        ]
                    }
                ]
            }
        else:
            # SHARED MODE: Anyone's Data + Not Expired
            where_filter = {
                 "$or": [
                    {"expiry_timestamp": {"$eq": -1.0}},
                    {"expiry_timestamp": {"$gt": current_time}}
                ]
            }
        
        # Run the search
        return self.collection.query(
            query_texts=[query_text],
            n_results=k,
            where=where_filter,
            include=['metadatas', 'documents', 'distances']
        )
    
    def rank_results(self, results: Dict[str, Any]) -> List[tuple]:
        """
        The Judge. Decides which memories are actually the best.
        
        It doesn't just look at text match ("Similarity").
        It also looks at:
        - Importance ("Is this a golden nugget?")
        - Freshness ("Is this news or history?")
        """
        # Balancing Act
        w_sim = 0.7      # 70% Match Quality
        w_ref = 0.2      # 20% Importance
        w_recency = 0.1  # 10% Freshness
        
        ranked_memories = []
        
        # Unpack the messy database response
        ids = results['ids'][0] if isinstance(results['ids'][0], list) else results['ids']
        distances = results['distances'][0] if isinstance(results['distances'][0], list) else results['distances']
        metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
        
        documents = results.get('documents', [[]])[0] if isinstance(results.get('documents', [[]])[0], list) else results.get('documents', [])
        
        current_time = datetime.now().timestamp()
        
        for i, (memory_id, dist, meta) in enumerate(zip(ids, distances, metadatas)):
            # Combine text and data
            if documents and i < len(documents):
                meta = meta.copy() if meta else {}
                meta['content'] = documents[i]
            
            # Distance is "How far away is it?". We want Similarity (1 - Distance).
            similarity = 1.0 - dist
            
            # Normalize Importance to 0.0 - 1.0 (It's usually 0-100)
            reflection_norm = meta.get('reflection_score', 50.0) / 100.0
            
            # Decay Score: Newer is better, but it drops off slowly.
            age_hours = (current_time - meta.get('created_at', current_time)) / 3600.0
            recency_score = 1.0 / (1.0 + 0.1 * age_hours)
            
            # Final Weighted Score
            final_score = (w_sim * similarity) + \
                          (w_ref * reflection_norm) + \
                          (w_recency * recency_score)
            
            ranked_memories.append((memory_id, final_score, meta))
        
        # Sort so the winners are at the top
        return sorted(ranked_memories, key=lambda x: x[1], reverse=True)
    
    def update_access_metadata(self, memory_id: str) -> None:
        """
        "I just remembered this!"
        Updates the 'last_seen' timestamp so we know this memory is still useful.
        """
        # Find it first
        result = self.collection.get(ids=[memory_id])
        
        if not result['ids']:
            return  # Ghost memory?
        
        current_metadata = result['metadatas'][0]
        
        # Update usage stats
        current_metadata['last_accessed'] = datetime.now().timestamp()
        current_metadata['access_count'] = current_metadata.get('access_count', 0) + 1
        
        # Save changes
        self.collection.update(
            ids=[memory_id],
            metadatas=[current_metadata]
        )
    
    def delete_memory(self, memory_id: str) -> None:
        """Erase a memory forever."""
        self.collection.delete(ids=[memory_id])
    
    def get_collection_stats(self) -> dict:
        """Quick check: How many memories do we have?"""
        count = self.collection.count()
        return {
            'total_memories': count,
            'collection_name': self.collection.name
        }
    
    def update_memory_metadata(self, memory_id: str, updates: dict) -> None:
        """
        Atomic update for memory details (like changing a Score).
        """
        self.collection.update(
            ids=[memory_id],
            metadatas=[updates]
        )
    
    def get_all_memories_for_user(self, user_id: str) -> List['MemoryChunk']:
        """
        Dump everything this user knows.
        Warning: Can be big!
        """
        from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
        
        results = self.collection.get(
            where={"user_id": {"$eq": user_id}},
            include=['metadatas', 'documents']
        )
        
        chunks = []
        for doc_id, metadata, document in zip(
            results['ids'],
            results['metadatas'],
            results['documents']
        ):
            # Reconstruct the object from raw data
            chunk = MemoryChunk(
                id=doc_id,
                content=document,
                type=MemoryType(metadata.get('type', 'episodic')),
                policy=PolicyClass(metadata.get('policy', 'ephemeral')),
                source_session_id=metadata.get('source_session_id', ''),
                user_id=metadata['user_id'],
                tags=metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                reflection_score=metadata.get('reflection_score', 50.0),
                created_at=metadata.get('created_at', 0.0),
                last_accessed=metadata.get('last_accessed', metadata.get('created_at', 0.0)),
                ttl_seconds=metadata.get('ttl_seconds')
            )
            chunk.access_count = metadata.get('access_count', 0)
            chunks.append(chunk)
        
        return chunks
    
    def get_memories_by_type(
        self, 
        user_id: str, 
        memory_type: 'MemoryType'
    ) -> List['MemoryChunk']:
        """
        Get only specific memories (e.g., "Just give me the Facts").
        """
        from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
        
        results = self.collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"type": {"$eq": memory_type.value}}
                ]
            },
            include=['metadatas', 'documents']
        )
        
        chunks = []
        for doc_id, metadata, document in zip(
            results['ids'],
            results['metadatas'],
            results['documents']
        ):
            chunk = MemoryChunk(
                id=doc_id,
                content=document,
                type=MemoryType(metadata.get('type', 'episodic')),
                policy=PolicyClass(metadata.get('policy', 'ephemeral')),
                source_session_id=metadata.get('source_session_id', ''),
                user_id=metadata['user_id'],
                tags=metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                reflection_score=metadata.get('reflection_score', 50.0),
                created_at=metadata.get('created_at', 0.0),
                last_accessed=metadata.get('last_accessed', metadata.get('created_at', 0.0)),
                ttl_seconds=metadata.get('ttl_seconds')
            )
            chunk.access_count = metadata.get('access_count', 0)
            chunks.append(chunk)
        
        return chunks

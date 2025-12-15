"""
Week 2 Day 4: NetworkX Graph Memory Overlay
============================================
Captures semantic relationships that vector similarity often misses.
Implements multi-session memory graph for entity relationships.
"""

import networkx as nx
import pickle
from typing import List, Tuple, Optional, Set
from pathlib import Path


class GraphMemory:
    """
    Graph-based memory overlay for capturing structured relationships.
    
    Uses NetworkX MultiDiGraph to store semantic triples (subject, predicate, object)
    and link them back to vector store chunks via UUIDs.
    
    Key Features:
    - Handles relationships like "Alice manages Bob" that vectors miss
    - BFS traversal for context expansion
    - Persistent storage via pickle
    - Links to ChromaDB documents for hybrid retrieval
    
    Reference: [7] NetworkX for graph operations
    Reference: [15] UUID linking between graph and vector store
    Reference: [1] Multi-session memory graph for prototype extensions
    """
    
    def __init__(self):
        """Initialize an empty directed multi-graph for semantic triples."""
        # MultiDiGraph allows multiple edges between same node pair with different relations
        self.graph = nx.MultiDiGraph()
    
    def add_relation(
        self, 
        subject: str, 
        predicate: str, 
        object_entity: str, 
        source_chunk_id: str
    ) -> None:
        """
        Add a semantic triple to the graph.
        
        Links back to the vector chunk source via ID for hybrid retrieval.
        This allows us to traverse the graph and retrieve associated documents.
        
        Args:
            subject: The source entity (e.g., "Alice")
            predicate: The relationship type (e.g., "manages", "knows", "created")
            object_entity: The target entity (e.g., "Bob")
            source_chunk_id: UUID of the MemoryChunk in ChromaDB that contains this relation
        
        Example:
            graph.add_relation("Alice", "manages", "Bob", "chunk-123")
            graph.add_relation("Dog", "bites", "Man", "chunk-456")
        
        Note: MultiDiGraph allows multiple different relationships between same entities
        """
        self.graph.add_edge(
            subject,
            object_entity,
            relation=predicate,
            source_id=source_chunk_id
        )
    
    def get_related_entities(
        self, 
        entity: str, 
        depth: int = 1
    ) -> List[Tuple[str, str, dict]]:
        """
        Retrieve entities related to the given entity within specified depth.
        
        Uses BFS (Breadth-First Search) to explore the graph neighborhood,
        enabling context expansion for queries.
        
        Args:
            entity: The entity to start from (e.g., "Alice")
            depth: Maximum traversal depth (default: 1 for immediate neighbors)
        
        Returns:
            List of (source, target, edge_data) tuples containing:
            - source: Source entity
            - target: Target entity  
            - edge_data: Dict with 'relation' and 'source_id' keys
        
        Example:
            # If graph has: Alice --manages--> Bob --knows--> Charlie
            results = graph.get_related_entities("Alice", depth=2)
            # Returns: [("Alice", "Bob", {...}), ("Bob", "Charlie", {...})]
        """
        if entity not in self.graph:
            return []
        
        # Use BFS to find all edges within depth limit
        edges = []
        bfs_edges = nx.bfs_edges(self.graph, entity, depth_limit=depth)
        
        for source, target in bfs_edges:
            # Get all edges between this pair (MultiDiGraph can have multiple)
            edge_data_dict = self.graph.get_edge_data(source, target)
            
            # edge_data_dict is {key: {relation: ..., source_id: ...}, ...}
            for key, data in edge_data_dict.items():
                edges.append((source, target, data))
        
        return edges
    
    def get_outgoing_relations(self, entity: str) -> List[Tuple[str, str, str]]:
        """
        Get all outgoing relationships from an entity.
        
        Args:
            entity: The source entity
        
        Returns:
            List of (relation, target_entity, source_chunk_id) tuples
        """
        if entity not in self.graph:
            return []
        
        relations = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            relations.append((data['relation'], target, data['source_id']))
        
        return relations
    
    def get_incoming_relations(self, entity: str) -> List[Tuple[str, str, str]]:
        """
        Get all incoming relationships to an entity.
        
        Args:
            entity: The target entity
        
        Returns:
            List of (relation, source_entity, source_chunk_id) tuples
        """
        if entity not in self.graph:
            return []
        
        relations = []
        for source, _, data in self.graph.in_edges(entity, data=True):
            relations.append((data['relation'], source, data['source_id']))
        
        return relations
    
    def find_path(
        self, 
        source: str, 
        target: str, 
        max_depth: int = 3
    ) -> Optional[List[str]]:
        """
        Find shortest path between two entities.
        
        Args:
            source: Starting entity
            target: Goal entity
            max_depth: Maximum path length to search
        
        Returns:
            List of entities forming the path, or None if no path exists
        
        Example:
            path = graph.find_path("Alice", "Charlie")
            # Returns: ["Alice", "Bob", "Charlie"] if such path exists
        """
        try:
            # Get shortest path (by number of hops)
            path = nx.shortest_path(self.graph, source, target)
            if len(path) - 1 <= max_depth:  # path length is nodes - 1
                return path
            return None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_all_entities(self) -> Set[str]:
        """
        Get all entities (nodes) in the graph.
        
        Returns:
            Set of entity identifiers
        """
        return set(self.graph.nodes())
    
    def get_relation_count(self) -> int:
        """
        Get total number of relationships in the graph.
        
        Returns:
            Number of edges
        """
        return self.graph.number_of_edges()
    
    def save_graph(self, path: str) -> None:
        """
        Persist the graph to disk using pickle.
        
        Args:
            path: File path to save the graph (e.g., "./data/memory_graph.pkl")
        
        Example:
            graph.save_graph("./data/memory_graph.pkl")
        """
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load_graph(self, path: str) -> None:
        """
        Load a persisted graph from disk.
        
        Args:
            path: File path to load the graph from
        
        Example:
            graph = GraphMemory()
            graph.load_graph("./data/memory_graph.pkl")
        
        Raises:
            FileNotFoundError: If the graph file doesn't exist
        """
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.graph.clear()

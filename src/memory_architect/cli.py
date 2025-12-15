"""
Week 5 Day 4: CLI for Developers
==================================
Interactive command-line interface for debugging and testing.
"""

import sys
import time
from typing import Optional
from src.memory_architect.storage.vector_store import ChromaManager
from src.memory_architect.policy.privacy import PrivacyGuard
from src.memory_architect.core.schema import MemoryChunk, MemoryType, PolicyClass
from src.memory_architect.core.pruning import analyze_memory_health



class MemoryCLI:
    """Interactive CLI for Memory Architect debugging."""
    
    def __init__(self):
        """Initialize CLI with database and privacy guard."""
        print("Initializing Memory Architect CLI...")
        
        self.db = ChromaManager(persist_path="./data/.chroma")
        self.privacy_guard = PrivacyGuard()
        self.current_user = "cli_user"
        print("\nMemory Architect CLI v1.0")
        print("Type 'help' for available commands\n")
    
    def cmd_help(self):
        """Show available commands."""
        help_text = """
Available Commands:
==================
!mem query <text>        - Debug retrieval with scores
!mem add <text>          - Manually inject a memory
!mem list [limit]        - List recent memories
!policy show             - Display policy configuration
!stats                   - Show database statistics
!user <user_id>          - Switch current user (default: cli_user)
help                     - Show this help message
exit                     - Exit CLI

Examples:
---------
!mem query Python programming
!mem add User prefers Python for data science
!stats
        """
        print(help_text)
    
    def cmd_mem_query(self, text: str):
        """
        Debug memory retrieval.
        
        Shows:
        - Retrieved memories
        - Raw similarity scores
        - Reflection scores
        - Final hybrid scores
        """
        if not text:
            print("Error: Query text required")
            return
        
        print(f"\nQuerying: '{text}'")
        print("=" * 60)
        
        # Retrieve
        results = self.db.retrieve_candidates(
            query_text=text,
            user_id=self.current_user,
            k=5
        )
        
        # Rank
        ranked = self.db.rank_results(results)
        
        if not ranked:
            print("No memories found.")
            return
        
        print(f"Found {len(ranked)} memories:\n")
        
        for i, (memory_id, final_score, metadata) in enumerate(ranked, 1):
            content = metadata.get('content', '')[:80]
            reflection_score = metadata.get('reflection_score', 0)
            created = metadata.get('created_at', 0)
            age_hours = (time.time() - created) / 3600.0
            
            print(f"[{i}] Score: {final_score:.3f}")
            print(f"    Reflection: {reflection_score:.1f}/100")
            print(f"    Age: {age_hours:.1f} hours")
            print(f"    Content: {content}...")
            print()
    
    def cmd_mem_add(self, text: str):
        """Manually add a memory."""
        if not text:
            print("Error: Memory content required")
            return
        
        # Create memory
        chunk = MemoryChunk(
            content=text,
            type=MemoryType.EPISODIC,
            policy=PolicyClass.EPHEMERAL,
            source_session_id="cli_manual",
            user_id=self.current_user,
            tags=["cli"],
            reflection_score=50.0,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        # Store
        self.db.add_memory(chunk)
        
        print(f"✓ Added memory with ID: {chunk.id}")
        print(f"  User: {self.current_user}")
        print(f"  Content: {text[:100]}...")
    
    def cmd_mem_list(self, limit: str = "10"):
        """List recent memories for current user."""
        try:
            limit_int = int(limit)
        except ValueError:
            limit_int = 10
        
        memories = self.db.get_all_memories_for_user(self.current_user)
        
        if not memories:
            print(f"No memories found for user: {self.current_user}")
            return
        
        # Sort by creation time (newest first)
        memories.sort(key=lambda x: x.created_at, reverse=True)
        
        print(f"\nRecent Memories for {self.current_user}:")
        print("=" * 60)
        
        for i, mem in enumerate(memories[:limit_int], 1):
            age_hours = (time.time() - mem.created_at) / 3600.0
            print(f"[{i}] {mem.type.value} | Score: {mem.reflection_score:.1f}")
            print(f"    Age: {age_hours:.1f}h | {mem.content[:70]}...")
            print()
    
    def cmd_policy_show(self):
        """Display policy configuration."""
        print("\nPolicy Configuration:")
        print("=" * 60)
        
        print("\nPII Detection:")
        print("  Status: Enabled")
        print("  Entities: EMAIL, PHONE, CREDIT_CARD, SSN, PERSON")
        
        print("\nTTL Policies:")
        print("  EPHEMERAL: 7 days")
        print("  CANONICAL: No expiry")
        print("  PROCEDURAL: 30 days")
        
        print("\nReflection:")
        print("  Default score: 50.0")
        print("  Reinforcement: +10.0 per use")
        print("  Decay: Exponential (stability=2.0)")
        
        print("\nGarbage Collection:")
        print("  Low-value threshold: 15.0")
        print("  Minimum age: 1 week")
        print("  Canonical protection: ON")
    
    def cmd_stats(self):
        """Show database statistics."""
        print("\nDatabase Statistics:")
        print("=" * 60)
        
        # Collection stats
        stats = self.db.get_collection_stats()
        print(f"Total memories: {stats.get('total_memories', 0):,}")
        print(f"Collection: {stats.get('collection_name', 'unknown')}")
        
        # User-specific stats
        user_memories = self.db.get_all_memories_for_user(self.current_user)
        print(f"\nCurrent user ({self.current_user}):")
        print(f"  Memories: {len(user_memories)}")
        
        if user_memories:
            avg_score = sum(m.reflection_score for m in user_memories) / len(user_memories)
            print(f"  Average score: {avg_score:.1f}")
            
            # Memory health
            health = analyze_memory_health(self.db, self.current_user)
            print(f"  Low-value: {health['low_score_count']}")
            print(f"  Canonical: {health['canonical_count']}")
    
    def cmd_user(self, user_id: str):
        """Switch current user."""
        if not user_id:
            print(f"Current user: {self.current_user}")
            return
        
        self.current_user = user_id
        print(f"Switched to user: {user_id}")
    
    def parse_command(self, line: str):
        """Parse and execute command."""
        line = line.strip()
        
        if not line:
            return
        
        if line.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            sys.exit(0)
        
        if line.lower() == 'help':
            self.cmd_help()
            return
        
        # Parse !mem commands
        if line.startswith('!mem '):
            parts = line[5:].split(maxsplit=1)
            if not parts:
                print("Error: Invalid !mem command")
                return
            
            subcmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if subcmd == 'query':
                self.cmd_mem_query(arg)
            elif subcmd == 'add':
                self.cmd_mem_add(arg)
            elif subcmd == 'list':
                self.cmd_mem_list(arg)
            else:
                print(f"Error: Unknown !mem subcommand: {subcmd}")
        
        elif line.startswith('!policy '):
            parts = line[8:].split()
            if parts and parts[0].lower() == 'show':
                self.cmd_policy_show()
            else:
                print("Error: Invalid !policy command (try: !policy show)")
        
        elif line.startswith('!stats'):
            self.cmd_stats()
        
        elif line.startswith('!user '):
            user_id = line[6:].strip()
            self.cmd_user(user_id)
        
        else:
            print(f"Unknown command: {line}")
            print("Type 'help' for available commands")
    
    def run(self):
        """Main CLI loop."""
        while True:
            try:
                line = input(f"{self.current_user}> ")
                self.parse_command(line)
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Entry point for CLI."""
    cli = MemoryCLI()
    cli.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Start the Memory Architect API server with proper configuration.
"""

import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if __name__ == "__main__":
    print("Starting Memory Architect API...")
    print("Server will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        "memory_architect.server.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

import json
import logging
from io import StringIO
from src.memory_architect.utils.logger import setup_logger, JsonFormatter

def test_json_logging():
    # Capture stdout
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    
    logger = logging.getLogger("test_logger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Log a message with extra fields
    extra = {"event_type": "retrieval", "memory_id": "mem-123", "latency_ms": 45.6}
    logger.info("Test message", extra=extra)
    
    # Get output
    output = stream.getvalue().strip()
    print(f"Log Output: {output}")
    
    # Verify JSON
    try:
        log_record = json.loads(output)
        print("JSON parsing successful.")
    except json.JSONDecodeError:
        print("JSON parsing failed.")
        exit(1)
        
    # Verify fields
    assert log_record["message"] == "Test message"
    assert log_record["level"] == "INFO"
    assert log_record["component"] == "test_logger"
    assert log_record["memory_event_type"] == "retrieval"
    assert log_record["memory_id"] == "mem-123"
    assert log_record["latency_ms"] == 45.6
    print("All assertions passed.")

if __name__ == "__main__":
    test_json_logging()

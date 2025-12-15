import logging
import sys
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "memory_event_type": getattr(record, "event_type", "system"),
            "memory_id": getattr(record, "memory_id", None),
            "latency_ms": getattr(record, "latency_ms", None)
        }
        return json.dumps(log_record)

def setup_logger(name: str):
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if setup_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

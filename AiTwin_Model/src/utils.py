# utils.py

import json
import os
import time
import yaml
import uuid
import aiofiles
from typing import Any

METRICS_PATH = "metrics.json"

def create_text_message(content: str, source: str, **kwargs) -> dict:
    """Creates a dictionary for a TextMessage with all required fields."""
    message = {
        "id": str(uuid.uuid4()),
        "source": source,
        "content": content,
        "timestamp": time.time(),
        "status": "sent",
        **kwargs
    }
    return message

def safe_model_dump(obj) -> dict:
    """Safely converts a Pydantic model or object to a JSON-serializable dictionary."""
    # Handles objects that might not be directly serializable (like Pydantic models)
    return json.loads(json.dumps(obj, default=str))

async def _record_metrics(data: dict):
    """Asynchronously records metrics data to the METRICS_PATH file."""
    if not os.path.exists(METRICS_PATH):
        async with aiofiles.open(METRICS_PATH, "w") as file:
            await file.write(json.dumps([]))

    async with aiofiles.open(METRICS_PATH, "r+") as file:
        contents = await file.read()
        metrics_list = json.loads(contents)
        metrics_list.append(data)
        await file.seek(0)
        await file.truncate()
        await file.write(json.dumps(metrics_list, indent=2, default=str))

def load_llm_config(filepath: str) -> dict:
    """Loads LLM configuration from a YAML file."""
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
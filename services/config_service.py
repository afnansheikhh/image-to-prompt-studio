"""
Configuration and API key persistence service.
"""

import os
import json
from typing import Dict, Any

CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".api_keys.json"))


class ConfigService:
    """Manages persistent API keys and provider preferences."""

    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """Load persistent configuration from file or environment."""
        cfg = {
            "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
            "groq_vision_model": "qwen/qwen3.8-27b"
        }

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        if saved.get("groq_api_key"):
                            cfg["groq_api_key"] = saved["groq_api_key"]
                        if saved.get("groq_vision_model"):
                            cfg["groq_vision_model"] = saved["groq_vision_model"]
            except Exception:
                pass

        return cfg

    @classmethod
    def save_config(cls, data: Dict[str, Any]) -> bool:
        """Save configuration dictionary to persistent json file."""
        try:
            current = cls.load_config()
            current.update(data)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
            return True
        except Exception:
            return False

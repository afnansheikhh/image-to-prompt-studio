"""
LLM Service for Local AI Prompt Enhancement via Ollama or local OpenAI-compatible endpoints.
Provides local model status checking, structured JSON prompting, and graceful fallback.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Tuple, Optional


class LocalLLMService:
    """Local LLM client for Ollama."""

    DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

    @classmethod
    def check_status(cls, endpoint: str = DEFAULT_ENDPOINT) -> Tuple[bool, List[str], str]:
        """
        Check if Ollama is running and retrieve list of available models.
        Returns: (is_available, model_names, status_message)
        """
        try:
            url = f"{endpoint.rstrip('/')}/api/tags"
            req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    if models:
                        return True, models, f"Connected ({len(models)} model{'s' if len(models) != 1 else ''} available)"
                    return True, [], "Ollama running (no models found, run 'ollama pull llama3')"
                return False, [], f"Ollama returned HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError) as e:
            return False, [], "Ollama not detected on http://127.0.0.1:11434"

    @classmethod
    def generate_enhanced_prompt(
        cls,
        base_caption: str,
        attributes: Dict[str, str],
        model_name: str = "llama3",
        creativity: float = 0.7,
        prompt_length: str = "Balanced",
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> Optional[Dict[str, Any]]:
        """
        Send structured scene attributes to local LLM and retrieve structured visual attributes & prompt.
        """
        system_instruction = (
            "You are an expert AI prompt engineer for image generation systems (Midjourney v6, SDXL, DALL-E 3, FLUX).\n"
            "Your task is to transform a base image caption and specified visual controls into an exquisite, highly detailed image generation prompt.\n"
            "RULES:\n"
            "1. PRESERVE SEMANTIC GROUNDING: Do NOT hallucinate or introduce non-existent key subjects, animals, or objects not supported by the base caption.\n"
            "2. Enhance aesthetic fidelity, textural nuances, lighting physics, camera lens characteristics, and composition.\n"
            "3. Output strictly valid JSON matching the exact schema requested with no markdown formatting or commentary outside the JSON."
        )

        user_prompt = f"""
Input Image Scene Data:
- Base Caption: "{base_caption}"
- Subject: {attributes.get('Subject', '')}
- Environment: {attributes.get('Environment', '')}
- Objects: {attributes.get('Objects', '')}
- Action/Pose: {attributes.get('Action / Pose', '')}
- Target Style: {attributes.get('Visual Style', '')}
- Lighting: {attributes.get('Lighting', '')}
- Mood: {attributes.get('Mood', '')}
- Color Palette: {attributes.get('Color Palette', '')}
- Composition: {attributes.get('Composition', '')}
- Camera/Lens: {attributes.get('Camera Perspective', '')}
- Detail Level: {attributes.get('Level of Detail', '')}
- Desired Length: {prompt_length}

Generate the final JSON object with this exact schema:
{{
  "prompt": "<detailed coherent image generation prompt combining the scene elements and visual styling>",
  "negative_prompt": "<comma-separated negative prompt tokens to avoid artifacts for this style>",
  "subject": "<refined subject description>",
  "environment": "<refined environmental setting>",
  "style": "<style description>",
  "lighting": "<lighting description>",
  "mood": "<mood description>",
  "composition": "<composition description>",
  "camera": "<camera/lens description>",
  "color_palette": "<color palette description>",
  "detail_level": "<detail level description>"
}}
"""

        payload = {
            "model": model_name,
            "prompt": f"{system_instruction}\n\n{user_prompt}",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": max(0.1, min(1.0, float(creativity))),
                "top_p": 0.9,
            }
        }

        try:
            url = f"{endpoint.rstrip('/')}/api/generate"
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as response:
                if response.status == 200:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    raw_text = resp_data.get("response", "").strip()
                    
                    # Parse JSON from response
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict) and "prompt" in parsed:
                        return parsed
        except Exception:
            return None

        return None

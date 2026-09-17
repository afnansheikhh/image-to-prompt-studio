import json
import io
import base64
import urllib.request
import urllib.error
import re
import time
from typing import Dict, Any, Optional, Tuple
from PIL import Image


class GroqService:
    """Service to communicate with Groq Cloud ultra-fast inference API."""

    CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    @classmethod
    def test_connection(cls, api_key: str) -> Tuple[bool, str]:
        """Verify if the provided Groq API key is active and valid with a lightweight ping."""
        if not api_key or len(api_key.strip()) < 10:
            return False, "API key cannot be empty or too short."

        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 2
        }

        headers = dict(cls.DEFAULT_HEADERS)
        headers["Authorization"] = f"Bearer {api_key.strip()}"

        try:
            req = urllib.request.Request(
                cls.CHAT_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status == 200:
                    return True, "🟢 Groq Vision API Connected & Verified! (Model: qwen/qwen3.8-27b)"
                else:
                    return False, f"Server returned HTTP status {resp.status}"
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            try:
                parsed_err = json.loads(err_msg)
                detail = parsed_err.get("error", {}).get("message", str(e))
            except Exception:
                detail = str(e)
            return False, f"🔴 Groq Error ({e.code}): {detail}"
        except Exception as e:
            return False, f"🔴 Connection Error: {str(e)}"

    @classmethod
    def _repair_and_parse_json(cls, text: str) -> Optional[Dict[str, Any]]:
        """Robustly parses and auto-repairs truncated or imperfect JSON output."""
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        start_idx = clean_text.find("{")
        if start_idx == -1:
            return None
        clean_text = clean_text[start_idx:]

        # 1. Direct JSON parse
        try:
            return json.loads(clean_text)
        except Exception:
            pass

        # 2. Regex search for complete JSON object
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass

        # 3. Progressive truncation & auto-closing of open braces / quotes
        s = clean_text
        for cut_point in range(len(s), max(0, len(s) - 500), -1):
            sub = s[:cut_point].rstrip(' \t\n\r,')
            quotes = len(re.findall(r'(?<!\\)"', sub))
            if quotes % 2 != 0:
                sub += '"'

            open_curly = sub.count("{") - sub.count("}")
            open_square = sub.count("[") - sub.count("]")

            if open_curly < 0 or open_square < 0:
                continue

            candidate = sub + ("]" * open_square) + ("}" * open_curly)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed
            except Exception:
                continue

        # 4. Regex key-value extraction fallback
        extracted = {}
        for key in ["image_type", "overall_description", "subjects", "objects", "environment", "text_content", "chart_data", "visual_style", "confidence"]:
            m = re.search(rf'"{key}"\s*:\s*([^,\n\}}]+)', clean_text)
            if m:
                extracted[key] = m.group(1).strip().strip('"')
        if extracted:
            return extracted

        return None

    @classmethod
    def analyze_with_vision(
        cls,
        image: Image.Image,
        api_key: str,
        model: str = "qwen/qwen3.8-27b"
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Directly analyze image pixels with Groq multimodal vision models."""
        if not api_key or len(api_key.strip()) < 10:
            return None, "Groq API key cannot be empty or too short."

        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_b64}"

        system_prompt = """You are an expert multimodal visual intelligence analyst and generative AI prompt engineer.
Analyze the provided image with forensic precision. Your output must strictly reflect WHAT IS VISUALLY PRESENT in the image without inventing details.

STRICT DOMAIN RULES:
1. If the image is a CHART / GRAPH / INFOGRAPHIC / DIAGRAM:
   - "image_type" MUST be "Chart / graph / visualization".
   - Transcribe the exact title, axis labels, legend names, data series, values, and bar/slice colors into "text_content" and "chart_data".
   - DO NOT hallucinate camera lenses (e.g. 50mm, f/1.4), shallow depth of field, outdoor nature, or human portrait elements.
2. If the image is a PERSON / PORTRAIT:
   - "image_type" MUST be "Person / portrait".
   - Identify precise subject details: gender, approximate age, expression, pose, clothing items, colors, patterns.
3. If the image is a UI SCREENSHOT / DIGITAL INTERFACE:
   - "image_type" MUST be "Screenshot / UI / digital interface".
   - Transcribe all visible navigation menus, buttons, titles, cards into "text_content" and "ui_elements".
4. If the image is a DOCUMENT / TEXT-HEAVY IMAGE:
   - "image_type" MUST be "Document / text-heavy image".
   - Transcribe headings, paragraph structure, and layout into "text_content" and "document_elements".

Keep entries concise and dense. Return a single valid JSON object with:
- "image_type": "Chart / graph / visualization | Person / portrait | Photograph / natural scene | Screenshot / UI / digital interface | Document / text-heavy image | Other"
- "overall_description": "Accurate 1-2 paragraph visual scene breakdown."
- "subjects": ["List of main subjects"]
- "objects": ["List of detected specific objects, elements, bars, components"]
- "environment": "Forensic description of background, coordinate canvas"
- "actions": ["List of actions / dynamic poses"]
- "spatial_relationships": ["Spatial positioning of elements"]
- "clothing": ["Apparel / accessories"]
- "colors": {"dominant": [], "secondary": [], "background": [], "temperature": "warm | cool | neutral"}
- "lighting": "Lighting description"
- "camera": {"viewpoint": "eye-level | low-angle | top-down", "shot_type": "close-up | medium | full | vector flat", "depth_of_field": "deep focus | shallow focus | flat 2D"}
- "text_content": ["List of all visible OCR text, labels, numbers found in image"]
- "chart_data": {"chart_type": "", "title": "", "x_axis": "", "y_axis": "", "series": []}
- "ui_elements": ["List of UI elements, buttons, cards if UI"]
- "visual_style": "Photorealistic | Flat 2D Vector | Digital UI | 3D Render",
- "confidence": 0.98
"""

        vision_models = [model, "qwen/qwen3.8-27b"]
        unique_vision_models = []
        for vm in vision_models:
            if vm and vm not in unique_vision_models:
                unique_vision_models.append(vm)

        last_error = "Unknown error"
        for vm_name in unique_vision_models:
            payload = {
                "model": vm_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt + "\nReturn ONLY the JSON object."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 800,
            }

            headers = dict(cls.DEFAULT_HEADERS)
            headers["Authorization"] = f"Bearer {api_key.strip()}"

            # Auto-retry on 429 rate limit (up to 3 attempts with smart backoff)
            for attempt in range(4):
                try:
                    req = urllib.request.Request(
                        cls.CHAT_ENDPOINT,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                    )
                    with urllib.request.urlopen(req, timeout=35.0) as resp:
                        if resp.status == 200:
                            resp_json = json.loads(resp.read().decode("utf-8"))
                            content = resp_json["choices"][0]["message"]["content"]
                            parsed = cls._repair_and_parse_json(content)
                            if parsed:
                                return parsed, ""
                            else:
                                last_error = "Failed to parse structured JSON from Vision model response."
                except urllib.error.HTTPError as e:
                    err_msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
                    try:
                        parsed_err = json.loads(err_msg)
                        last_error = parsed_err.get("error", {}).get("message", str(e))
                    except Exception:
                        last_error = f"HTTP {e.code}: {err_msg[:120]}"

                    if e.code == 429 and attempt < 3:
                        wait_seconds = 1.2 + (attempt * 1.0)
                        match = re.search(r"try again in ([0-9\.]+)m?s", last_error, re.IGNORECASE)
                        if match:
                            val = float(match.group(1))
                            if "ms" in match.group(0).lower():
                                wait_seconds = max(0.8, (val / 1000.0) + 0.4)
                            else:
                                wait_seconds = max(0.8, val + 0.4)
                        time.sleep(wait_seconds)
                        continue
                    else:
                        break
                except Exception as e:
                    last_error = str(e)
                    break

        return None, last_error

    @classmethod
    def enhance_prompt(
        cls,
        base_caption: str,
        attributes: Dict[str, Any],
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        image_type: str = "Photograph / natural scene",
    ) -> Optional[Dict[str, Any]]:
        """Synthesize high-fidelity generative prompts using Groq LLM with automatic model fallback."""
        if not api_key or len(api_key.strip()) < 10:
            return None

        candidate_models = [
            model,
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "llama3-8b-8192",
        ]
        # Deduplicate preserving order
        unique_models = []
        for m in candidate_models:
            if m and m not in unique_models:
                unique_models.append(m)

        system_prompt = f"""You are an elite generative AI prompt engineer and visual intelligence expert.
Your goal is to transform the provided image description and visual attributes into an exquisite, production-grade generative prompt for modern AI image generators (Midjourney v6, FLUX, Stable Diffusion SDXL).

Target Domain: {image_type}

RULES:
1. If the image is a Chart, Graph, UI Screenshot, or Document: Strictly generate clean vector design/UI/PDF layout prompts without photographic camera buzzwords (no 50mm, no f/1.4, no bokeh).
2. If the image is a Photograph or Natural scene: Generate rich photographic prompts with lighting physics, camera perspectives, and texture details.

Return a STRICT JSON object with these exact keys:
{{
  "detailed_description": "<grounded 2-paragraph visual scene description>",
  "master_prompt": "<exquisite masterwork prompt for FLUX/Midjourney/SDXL>",
  "midjourney_prompt": "<formatted with /imagine prompt: ... --v 6.0 --style raw --ar 16:9>",
  "stable_diffusion_prompt": "<weighted tokens for SDXL>",
  "negative_prompt": "<targeted negative prompt>",
  "attributes": {{
    "Domain": "{image_type}",
    "Subject": "...",
    "Environment / Context": "...",
    "Lighting / Palette": "...",
    "Composition / Layout": "...",
    "Level of Detail": "..."
  }}
}}
"""

        user_content = f"Base Image Description: {base_caption}\nStructured Visual Attributes: {json.dumps(attributes)}"

        for m_name in unique_models:
            payload = {
                "model": m_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            }

            try:
                req = urllib.request.Request(
                    cls.CHAT_ENDPOINT,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key.strip()}",
                        "Content-Type": "application/json",
                        "User-Agent": "PromptStudio/1.0",
                    },
                )
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    if resp.status == 200:
                        resp_json = json.loads(resp.read().decode("utf-8"))
                        content = resp_json["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                        return parsed
            except Exception:
                continue

        return None

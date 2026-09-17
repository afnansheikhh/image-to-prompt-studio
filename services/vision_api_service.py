"""
Vision API Service for Detailed Image Scene Understanding and Prompt Synthesis.
"""

import json
import base64
import urllib.request
import urllib.error
import re
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import io


class VisionAPIService:
    """Calls Gemini Vision API for deep image understanding and prompt generation."""

    _cached_models: List[str] = []

    @classmethod
    def get_supported_models(cls, api_key: str) -> List[str]:
        """Fetch list of active generation models for this API key."""
        if cls._cached_models:
            return cls._cached_models

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key.strip()}"
        try:
            req = urllib.request.Request(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
                }
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status == 200:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    models = resp_json.get("models", [])
                    gen_models = [
                        m["name"].replace("models/", "")
                        for m in models
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    if gen_models:
                        cls._cached_models = gen_models
                        return gen_models
        except Exception:
            pass

        return [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash-latest"
        ]

    @classmethod
    def test_connection(cls, api_key: str) -> Tuple[bool, str]:
        """Verify if the provided Gemini API key is active and valid using ModelService.ListModels."""
        if not api_key or len(api_key.strip()) < 10:
            return False, "API key cannot be empty or too short."

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key.strip()}"
        try:
            req = urllib.request.Request(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
                }
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status == 200:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    models = resp_json.get("models", [])
                    generate_models = [m["name"].replace("models/", "") for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
                    cls._cached_models = generate_models
                    summary = ", ".join(generate_models[:2]) if generate_models else "Ready"
                    return True, f"🟢 Gemini API Connected & Verified! ({summary})"
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            try:
                parsed = json.loads(err_msg)
                detail = parsed.get("error", {}).get("message", str(e))
            except Exception:
                detail = str(e)
            return False, f"🔴 Gemini Error ({e.code}): {detail}"
        except Exception as e:
            return False, f"🔴 Connection Error: {str(e)}"

        return False, "🔴 Unable to verify Gemini API connection."

    @classmethod
    def analyze_with_gemini(
        cls,
        image: Image.Image,
        api_key: str
    ) -> Optional[Dict[str, Any]]:
        """Exhaustively analyze image pixels and generate rich prompts."""
        if not api_key or len(api_key.strip()) < 10:
            return None

        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        system_instruction = """You are an expert multimodal visual intelligence analyst and generative AI prompt engineer.
Analyze the provided image with forensic precision. Your output must strictly reflect WHAT IS VISUALLY PRESENT in the image without inventing or hallucinating details.

STRICT DOMAIN RULES:
1. If the image is a CHART / GRAPH / INFOGRAPHIC / DIAGRAM:
   - "image_type" MUST be "Chart / graph / visualization".
   - Transcribe the exact title, axis labels, legend names, data series, values, and bar/slice colors into "text_content" and "chart_data".
   - DO NOT hallucinate camera lenses (e.g. 50mm, f/1.4), shallow depth of field, outdoor nature, or human portrait elements.
2. If the image is a PERSON / PORTRAIT:
   - "image_type" MUST be "Person / portrait".
   - Identify precise subject details: gender, approximate age, expression, pose, facial hair, eye direction.
   - Describe exact clothing items, colors, patterns, and accessories (e.g. sunglasses, hats, jewelry).
   - Describe the exact background (e.g. waterfall, studio backdrop, cityscape, forest).
3. If the image is a UI SCREENSHOT / DIGITAL INTERFACE:
   - "image_type" MUST be "Screenshot / UI / digital interface".
   - Transcribe all visible navigation menus, buttons, titles, cards, and theme colors into "text_content" and "ui_elements".
4. If the image is a DOCUMENT / TEXT-HEAVY IMAGE:
   - "image_type" MUST be "Document / text-heavy image".
   - Transcribe headings, paragraph structure, and layout into "text_content" and "document_elements".

You MUST return a single, valid JSON object strictly matching this schema:
{
  "image_type": "Chart / graph / visualization | Person / portrait | Photograph / natural scene | Screenshot / UI / digital interface | Document / text-heavy image | Product / object | Architecture / interior | Artwork / illustration | Other",
  "overall_description": "Comprehensive, forensic 2-3 paragraph breakdown of everything in the image.",
  "subjects": ["List of main subjects with descriptive details"],
  "objects": ["List of detected specific objects, elements, props, visual components"],
  "environment": "Forensic description of the background, setting, surroundings, or coordinate canvas",
  "actions": ["List of actions, interactions, poses, or dynamic motions"],
  "spatial_relationships": ["List describing spatial positioning of subjects and objects"],
  "people": [{"gender": "", "age_group": "", "expression": "", "pose": ""}],
  "clothing": ["Exact clothing pieces, apparel colors, and accessories"],
  "colors": {
    "dominant": ["List of main colors"],
    "secondary": ["List of secondary/accent colors"],
    "background": ["List of background colors"],
    "temperature": "warm | cool | neutral"
  },
  "lighting": "Description of lighting type, direction, intensity, and shadow characteristics",
  "camera": {
    "viewpoint": "eye-level | low-angle | high-angle | bird's-eye | top-down",
    "angle": "straight-on | 45-degree | profile | wide-angle",
    "shot_type": "close-up | medium shot | full shot | extreme close-up | panoramic | vector flat",
    "depth_of_field": "deep focus | shallow focus | flat 2D",
    "focal_perspective": "natural perspective | wide perspective | telephoto compression | orthographic flat"
  },
  "composition": ["List of composition rules, framing, rule-of-thirds, symmetry, margins"],
  "text_content": ["List of all exact text strings, titles, labels, numbers, and OCR transcriptions found in image"],
  "chart_data": {
    "chart_type": "",
    "title": "",
    "x_axis": "",
    "y_axis": "",
    "legend": [],
    "series": []
  },
  "ui_elements": ["List of UI elements, buttons, cards, headers if UI"],
  "document_elements": ["List of document headers, columns, margins if document"],
  "visual_style": "Photorealistic | Flat 2D Vector | Digital UI | 3D Render | Oil Painting | Watercolor | Technical Diagram",
  "confidence": 0.95
}
"""

        payload = {
            "contents": [{
                "parts": [
                    {"text": system_instruction},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        # Dynamically get available models
        candidate_models = cls.get_supported_models(api_key)
        preferred_order = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"]
        for pref in reversed(preferred_order):
            if pref in candidate_models:
                candidate_models.remove(pref)
                candidate_models.insert(0, pref)

        for model in candidate_models[:5]:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
                    }
                )
                with urllib.request.urlopen(req, timeout=25.0) as resp:
                    if resp.status == 200:
                        resp_json = json.loads(resp.read().decode("utf-8"))
                        text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                        
                        clean_text = text.strip()
                        if clean_text.startswith("```json"):
                            clean_text = clean_text[7:]
                        if clean_text.startswith("```"):
                            clean_text = clean_text[3:]
                        if clean_text.endswith("```"):
                            clean_text = clean_text[:-3]
                        clean_text = clean_text.strip()

                        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
                        if json_match:
                            clean_text = json_match.group(0)

                        parsed = json.loads(clean_text)
                        return parsed
            except Exception:
                # If response_mime_type is rejected by older endpoint/model, retry without it
                try:
                    payload_fallback = {
                        "contents": [{
                            "parts": [
                                {"text": system_instruction},
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": img_b64
                                    }
                                }
                            ]
                        }],
                        "generationConfig": {
                            "temperature": 0.1,
                        }
                    }
                    req_fallback = urllib.request.Request(
                        endpoint,
                        data=json.dumps(payload_fallback).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
                        }
                    )
                    with urllib.request.urlopen(req_fallback, timeout=25.0) as resp:
                        if resp.status == 200:
                            resp_json = json.loads(resp.read().decode("utf-8"))
                            text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                            clean_text = text.strip()
                            if clean_text.startswith("```json"):
                                clean_text = clean_text[7:]
                            if clean_text.startswith("```"):
                                clean_text = clean_text[3:]
                            if clean_text.endswith("```"):
                                clean_text = clean_text[:-3]
                            clean_text = clean_text.strip()

                            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
                            if json_match:
                                clean_text = json_match.group(0)

                            parsed = json.loads(clean_text)
                            return parsed
                except Exception:
                    continue

        return None

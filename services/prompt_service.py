"""
Prompt Intelligence Engine:
Parses visual semantics and grounds prompt generation by image domain (Chart, UI, Document, Photo, etc.).
Prevents photographic hallucinations on analytical / digital images while retaining rich photographic synthesis for real-world photos.
"""

import re
import random
from typing import Dict, Any, Optional
from services.prompt_templates import (
    STYLE_PRESETS,
    LIGHTING_PRESETS,
    CAMERA_PRESETS,
    COMPOSITION_PRESETS,
    MOOD_PRESETS,
    COLOR_PRESETS,
    DETAIL_PRESETS,
)
from services.llm_service import LocalLLMService


class PromptIntelligenceEngine:
    """Type-routed prompt intelligence engine supporting Template and Local AI generation."""

    SUBJECT_ENHANCERS = {
        "dog": ["an expressive dog with detailed fur texture", "a lively canine subject with glistening eyes", "a playful dog captured in high-detail motion"],
        "cat": ["an elegant cat with sharp feline gaze", "a curious cat with velvety fur and fine whiskers", "a graceful feline posing elegantly"],
        "man": ["a charismatic man with distinct facial features", "a poised male figure with realistic skin tones", "a stylish man in authentic attire"],
        "woman": ["a striking woman with expressive eyes and graceful posture", "a confident female subject with natural radiant skin", "an elegant woman in detailed clothing"],
        "boy": ["a spirited young boy full of life", "a cheerful boy with genuine expression", "a young boy in dynamic motion"],
        "girl": ["a young girl with expressive joyful eyes", "a charming girl with finely detailed features", "a young female subject captured naturally"],
        "person": ["a compelling human subject with authentic presence", "a distinctive person captured in a candid moment", "a well-defined figure in natural clothing"],
        "people": ["a vibrant group of individuals interacting naturally", "a lively crowd with authentic human expressions", "a gathering of people with distinct personalities"],
        "child": ["a joyful child with innocent expression", "a dynamic young child captured candidly", "an expressive child in clear focus"],
        "car": ["a sleek automobile with glossy reflections and metallic paint", "a finely crafted vehicle on the open road", "a high-performance car with aerodynamic lines"],
        "bike": ["a detailed bicycle with mechanical precision", "a sleek bicycle resting in the scene", "a stylish bike with metallic accents"],
        "motorcycle": ["a custom motorcycle with polished chrome and raw power", "a rugged motorcycle with detailed mechanical components"],
        "bird": ["a majestic bird with intricate feather plumage", "a graceful bird in crisp avian focus"],
        "horse": ["a noble horse with muscular definition and flowing mane", "a majestic equestrian subject in full vitality"],
    }

    ENVIRONMENT_ENHANCERS = {
        "beach": "a sun-drenched coastal shoreline with gentle ocean waves, wet sand reflections, and coastal sea spray",
        "water": "a pristine body of crystal-clear water with realistic ripples and refractive caustic reflections",
        "ocean": "an expansive deep blue ocean with rolling white-capped waves and distant sea horizon",
        "river": "a flowing winding river with glistening water surface and lush riverbank vegetation",
        "lake": "a calm reflective mountain lake with mirror-like glassy water and scenic backdrop",
        "pool": "a shimmering turquoise swimming pool with bright underwater caustic light patterns",
        "street": "a bustling urban street with textured pavement, modern storefronts, and atmospheric city depth",
        "city": "a sprawling modern metropolis with towering architecture, glass skyscrapers, and lively urban energy",
        "road": "an expansive asphalt roadway stretching into the horizon under dramatic open skies",
        "park": "a picturesque landscaped green park with dappled sunlight through tree canopies",
        "field": "an open rolling grass field swaying in the gentle breeze under a wide sky",
        "grass": "a lush emerald green meadow with dew-kissed blades of grass and organic ground texture",
        "forest": "a dense enchanted forest with towering ancient trees, mossy trunks, and filtered ambient light",
        "woods": "a serene woodland grove with tall birch trees and dappled woodland shadows",
        "mountain": "majestic rugged mountain peaks rising in the background with crisp atmospheric haze",
        "snow": "a crisp winter landscape blanketed in pristine untouched powdery snow with sparkling crystals",
        "room": "a thoughtfully designed interior room with tasteful decor, soft furnishings, and cozy ambience",
        "bench": "a rustic park bench nestled under foliage in a peaceful scenic setting",
        "yard": "a spacious backyard surrounded by lush green flora and garden elements",
    }

    ACTION_ENHANCERS = {
        "running": "captured mid-stride in full kinetic motion with dynamic energy",
        "jumping": "suspended mid-air in an impressive athletic leap",
        "standing": "standing poised with grounded balance and commanding presence",
        "sitting": "resting comfortably in a relaxed and natural posture",
        "walking": "strolling purposefully with fluid natural movement",
        "playing": "engaged in spirited play and dynamic interaction",
        "looking": "gazing attentively with deep focused eye contact",
        "holding": "carefully clasping an object with tactile hand interaction",
        "riding": "riding smoothly with focused concentration and motion blur",
        "swimming": "gliding through refreshing water with dynamic aquatic splashes",
        "climbing": "scaling the terrain with athletic grip and strength",
        "smiling": "radiating a warm authentic smile and cheerful expression",
    }

    OBJECT_ENHANCERS = {
        "ball": "a brightly colored sports ball with realistic surface texture and spin",
        "frisbee": "a flying disc caught in mid-flight with aerodynamic clarity",
        "hat": "a stylish hat adding character and silhouette definition",
        "jacket": "a finely stitched jacket with tactile fabric weave and folds",
        "shirt": "casual garments with natural fabric draping and wrinkles",
        "sunglasses": "reflective dark sunglasses catching ambient lighting",
        "tree": "stately trees with intricate bark texture and lush leaf canopies",
        "flowers": "blooming wildflowers with delicate colorful petals",
    }

    @classmethod
    def parse_semantics(cls, caption: str) -> Dict[str, str]:
        """Extract semantic entities from photographic captions."""
        lowered = caption.lower()

        extracted_subject = None
        for key, enhancements in cls.SUBJECT_ENHANCERS.items():
            if key in lowered:
                extracted_subject = enhancements[0]
                break

        if not extracted_subject:
            if any(w in lowered for w in ["dog", "puppy", "canine", "cat", "kitten", "feline", "pet", "animal"]):
                extracted_subject = "an animal subject with authentic lifelike details"
            elif any(w in lowered for w in ["man", "guy", "male", "gentleman"]):
                extracted_subject = "a charismatic man with natural facial features and detailed clothing"
            elif any(w in lowered for w in ["woman", "lady", "female", "girl"]):
                extracted_subject = "a graceful woman with expressive features and poised presence"
            elif any(w in lowered for w in ["boy", "kid", "child", "children"]):
                extracted_subject = "a young subject with expressive natural emotion"
            elif any(w in lowered for w in ["person", "people", "figure"]):
                extracted_subject = "a compelling human subject with authentic presence"
            elif "subject" in lowered or len(caption.strip()) == 0:
                extracted_subject = "a compelling focal subject with natural presence"
            else:
                extracted_subject = f"a distinctive focal subject depicting {caption.strip()}"

        extracted_env = None
        for key, env_desc in cls.ENVIRONMENT_ENHANCERS.items():
            if key in lowered:
                extracted_env = env_desc
                break
        if not extracted_env:
            if any(w in lowered for w in ["wood", "woods", "forest", "tree", "trees", "jungle", "nature"]):
                extracted_env = "a scenic wooded grove surrounded by natural tree trunks and organic ground foliage"
            elif any(w in lowered for w in ["rock", "rocks", "stone", "boulder", "cliff"]):
                extracted_env = "a rugged rocky terrain with textured stone surfaces and natural outdoor ambience"
            elif any(w in lowered for w in ["city", "street", "urban", "building", "sidewalk"]):
                extracted_env = "a vibrant urban streetscape with modern architectural depth"
            else:
                extracted_env = "an authentic environmental setting with realistic atmospheric depth"

        extracted_action = None
        for key, act_desc in cls.ACTION_ENHANCERS.items():
            if key in lowered:
                extracted_action = act_desc
                break
        if not extracted_action:
            if any(w in lowered for w in ["sit", "sitting", "seated", "resting"]):
                extracted_action = "seated comfortably in a relaxed, poised posture"
            elif any(w in lowered for w in ["stand", "standing", "poised"]):
                extracted_action = "standing with grounded balance and commanding presence"
            elif any(w in lowered for w in ["pose", "posing", "look", "looking"]):
                extracted_action = "posing naturally with focused gaze"
            else:
                extracted_action = "captured in a natural spontaneous moment"

        found_objects = []
        for key, obj_desc in cls.OBJECT_ENHANCERS.items():
            if key in lowered:
                found_objects.append(obj_desc)
        extracted_objects = ", ".join(found_objects) if found_objects else "contextual props and organic scene elements"

        return {
            "subject": extracted_subject,
            "environment": extracted_env,
            "action": extracted_action,
            "objects": extracted_objects,
        }

    @classmethod
    def _build_chart_prompt(
        cls,
        vision_data: Dict[str, Any],
        prompt_length: str = "Balanced"
    ) -> Dict[str, Any]:
        """Build precise vector data visualization prompts without photographic buzzwords."""
        chart_subtype = vision_data.get("chart_subtype", "Bar Chart")
        title = vision_data.get("title", "Data Visualization")
        data_series = vision_data.get("data_series", "distinct quantitative metrics")
        labels = vision_data.get("category_labels", "category axes")
        palette_str = ", ".join(vision_data.get("color_palette", ["Navy Blue", "Cyan", "White"]))
        bg = vision_data.get("background", "Clean white background")

        if prompt_length == "Compact":
            main_prompt = (
                f"A modern 2D vector {chart_subtype.lower()} titled '{title}', displaying data values [{data_series}] "
                f"across [{labels}], clean flat SVG graphic design, {bg.lower()}, crisp typography."
            )
        elif prompt_length == "Extended":
            main_prompt = (
                f"A professional high-fidelity 2D vector data visualization infographic: a clean {chart_subtype.lower()} titled '{title}'. "
                f"Featuring crisp coordinate grid lines, prominent data columns displaying [{data_series}], categorized by [{labels}]. "
                f"Rendered in a polished business analytics palette of {palette_str} with {bg.lower()}. "
                f"Designed in modern Figma UI kit style, sharp vector edges, immaculate typography, zero noise, high contrast, SVG export quality."
            )
        else:  # Balanced
            main_prompt = (
                f"A clean modern vector {chart_subtype.lower()} titled '{title}', illustrating [{data_series}] for [{labels}]. "
                f"Crisp flat 2D graphic design, high contrast readable typography, formatted in {palette_str} against a {bg.lower()}, "
                f"minimalist vector illustration, SVG clean layout."
            )

        negative = (
            "photograph, realistic camera shot, 35mm, 50mm, f/1.4, bokeh, depth of field, real life photo, organic clutter, "
            "grain, blurry, artifacts, noisy texture, lens flare, distorted typography"
        )
        mj = f"/imagine prompt: {main_prompt} --v 6.0 --style raw --ar 16:9"
        sd = f"{main_prompt}, vector illustration, flat UI design, SVG graphic, crisp lines, clean typography, high resolution"

        attributes = {
            "Domain": "Chart / Graph / Visualization",
            "Chart Format": chart_subtype,
            "Chart Title": title,
            "Data Points": data_series,
            "Categories / Axes": labels,
            "Palette": palette_str,
            "Background": bg,
            "Layout Structure": "2D Cartesian coordinate grid",
            "Rendering Style": "Flat Vector Graphic / SVG",
            "Typography": "Clean Sans-Serif Data Labels",
            "Level of Detail": "High Precision Vector",
        }

        return {
            "main_prompt": main_prompt,
            "midjourney_prompt": mj,
            "stable_diffusion_prompt": sd,
            "dalle_prompt": main_prompt,
            "negative_prompt": negative,
            "structured_attributes": attributes,
        }

    @classmethod
    def _build_document_prompt(
        cls,
        vision_data: Dict[str, Any],
        prompt_length: str = "Balanced"
    ) -> Dict[str, Any]:
        """Build document and text layout prompts."""
        header = vision_data.get("header", "Executive Document")
        line_count = vision_data.get("line_count", 10)
        palette_str = ", ".join(vision_data.get("color_palette", ["Clean White", "Black", "Gray"]))

        main_prompt = (
            f"A clean editorial document page layout with prominent heading '{header}'. "
            f"Features structured typographic hierarchy, multi-paragraph columns ({line_count} lines), "
            f"crisp margins, elegant modern typography in {palette_str}, vector document graphic, editorial PDF layout."
        )

        negative = "photorealistic photo, natural outdoors, 50mm lens, bokeh, blurry, organic landscape, camera noise"
        mj = f"/imagine prompt: {main_prompt} --v 6.0 --style raw --ar 3:4"
        sd = f"{main_prompt}, high-resolution vector PDF layout, sharp typography, minimalist graphic"

        attributes = {
            "Domain": "Document / Text-Heavy Image",
            "Document Type": "Editorial Report / Document Page",
            "Header": header,
            "Estimated Text Density": f"{line_count} structured lines",
            "Palette": palette_str,
            "Layout": "Structured multi-column typographical layout",
            "Rendering Style": "Clean Editorial Vector PDF",
            "Typography": "High-contrast Editorial Serif & Sans",
            "Margins": "Balanced white space margins",
            "Level of Detail": "Sharp High-Resolution Text Grid",
            "Aesthetic": "Corporate & Editorial Modern",
        }

        return {
            "main_prompt": main_prompt,
            "midjourney_prompt": mj,
            "stable_diffusion_prompt": sd,
            "dalle_prompt": main_prompt,
            "negative_prompt": negative,
            "structured_attributes": attributes,
        }

    @classmethod
    def _build_ui_prompt(
        cls,
        vision_data: Dict[str, Any],
        prompt_length: str = "Balanced"
    ) -> Dict[str, Any]:
        """Build UI / UX screenshot design prompts."""
        theme = vision_data.get("theme", "Modern UI")
        nav = vision_data.get("visible_nav", "Navigation Controls")
        palette_str = ", ".join(vision_data.get("color_palette", ["Dark Gray", "Cyan", "White"]))

        main_prompt = (
            f"A sleek modern {theme.lower()} digital web application interface dashboard. "
            f"Featuring modular dashboard cards, navigation bar with [{nav}], responsive data widgets, "
            f"modern Figma UI/UX kit aesthetics, crisp iconography, {palette_str} color palette, glassmorphism elements."
        )

        negative = "photograph, real world photo, 50mm, bokeh, dirty, noisy, outdoors, nature scene, camera grain"
        mj = f"/imagine prompt: {main_prompt} --v 6.0 --style raw --ar 16:9"
        sd = f"{main_prompt}, UI kit, Dribbble trending, Figma UI design, clean vector components, 8k UI mockup"

        attributes = {
            "Domain": "Screenshot / UI / Digital Interface",
            "Interface Type": "Web / Desktop Application Dashboard",
            "Theme": theme,
            "UI Modules": "Metric cards, navigation bar, charts, action buttons",
            "Navigation Elements": nav,
            "Palette": palette_str,
            "Design Framework": "Modern Figma / Tailwind UI Design System",
            "Component Style": "Clean rounded card widgets with subtle drop shadows",
            "Typography": "Modern Inter / SF Pro System Typography",
            "Level of Detail": "Pixel-Perfect UI Mockup",
            "Aspect Ratio": "16:9 Widescreen Interface",
        }

        return {
            "main_prompt": main_prompt,
            "midjourney_prompt": mj,
            "stable_diffusion_prompt": sd,
            "dalle_prompt": main_prompt,
            "negative_prompt": negative,
            "structured_attributes": attributes,
        }

    @classmethod
    def synthesize_from_vision_json(
        cls,
        vision_json: Dict[str, Any],
        mode: str = "Exact Image Recreation",
        creative_controls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes high-fidelity generative prompts from the structured 17-field Vision API response.
        Supports:
          - Mode 1: "Exact Image Recreation" (Faithful forensic reproduction of visual content, style, text, lighting, camera).
          - Mode 2: "Creative Prompt" (Applies custom style, lighting, camera, mood while preserving detected subjects, objects & spatial layout).
        """
        image_type = vision_json.get("image_type", "Photograph / natural scene")
        overall_desc = vision_json.get("overall_description", "")
        subjects = vision_json.get("subjects", [])
        objects = vision_json.get("objects", [])
        environment = vision_json.get("environment", "")
        actions = vision_json.get("actions", [])
        spatial = vision_json.get("spatial_relationships", [])
        people = vision_json.get("people", [])
        clothing = vision_json.get("clothing", [])
        colors = vision_json.get("colors", {})
        lighting = vision_json.get("lighting", "")
        camera = vision_json.get("camera", {})
        composition = vision_json.get("composition", [])
        text_content = vision_json.get("text_content", [])
        chart_data = vision_json.get("chart_data", {})
        ui_elements = vision_json.get("ui_elements", [])
        doc_elements = vision_json.get("document_elements", [])
        style = vision_json.get("visual_style", "Photorealistic")
        confidence = vision_json.get("confidence", 0.95)

        subj_str = ", ".join(subjects) if isinstance(subjects, list) else str(subjects)
        obj_str = ", ".join(objects) if isinstance(objects, list) else str(objects)
        action_str = ", ".join(actions) if isinstance(actions, list) else str(actions)
        spatial_str = ", ".join(spatial) if isinstance(spatial, list) else str(spatial)
        text_str = ", ".join(text_content) if isinstance(text_content, list) else str(text_content)
        clothing_str = ", ".join(clothing) if isinstance(clothing, list) else str(clothing)
        
        dom_colors = colors.get("dominant", []) if isinstance(colors, dict) else []
        dom_colors_str = ", ".join(dom_colors) if isinstance(dom_colors, list) else str(dom_colors)
        color_temp = colors.get("temperature", "") if isinstance(colors, dict) else ""

        cam_shot = camera.get("shot_type", "") if isinstance(camera, dict) else ""
        cam_view = camera.get("viewpoint", "") if isinstance(camera, dict) else ""
        cam_angle = camera.get("angle", "") if isinstance(camera, dict) else ""
        cam_dof = camera.get("depth_of_field", "") if isinstance(camera, dict) else ""

        creative_controls = creative_controls or {}
        custom_style = creative_controls.get("style", "Auto / Original")
        custom_lighting = creative_controls.get("lighting", "Auto / Original")
        custom_camera = creative_controls.get("camera", "Auto / Original")
        custom_mood = creative_controls.get("mood", "Auto / Original")

        # Handle Chart / Graph / Infographic
        if image_type in ["Chart / graph / visualization", "Chart", "Infographic"]:
            c_title = chart_data.get("title", "") or (text_content[0] if text_content else "Data Visualization")
            c_type = chart_data.get("chart_type", "2D Vector Chart")
            c_series = chart_data.get("series", [])
            c_series_str = ", ".join(str(s) for s in c_series) if c_series else obj_str
            c_x = chart_data.get("x_axis", "")
            c_y = chart_data.get("y_axis", "")

            prompt_parts = [
                f"A clean modern 2D vector {c_type.lower()} infographic titled '{c_title}'",
                f"illustrating {c_series_str}" if c_series_str else "",
                f"with X-axis [{c_x}] and Y-axis [{c_y}]" if (c_x or c_y) else "",
                f"featuring crisp graphic elements: {obj_str}" if obj_str else "",
                f"rendered in a clear visual palette of {dom_colors_str}" if dom_colors_str else "",
                f"against {environment}" if environment else "clean minimal background",
                "clean flat SVG graphic design, high contrast legible typography, precise coordinate alignment, Figma UI kit vector quality"
            ]
            main_prompt = ". ".join(p for p in prompt_parts if p).strip() + "."
            negative_prompt = "photograph, realistic camera shot, 35mm, 50mm, f/1.4, bokeh, shallow depth of field, real life photo, human portrait, organic clutter, grain, blurry, camera noise"
            mj_prompt = f"/imagine prompt: {main_prompt} --v 6.0 --style raw --ar 16:9"
            sd_prompt = f"{main_prompt}, vector illustration, flat UI design, SVG graphic, crisp lines, clean typography, 8k resolution"

        # Handle UI / Digital Interface
        elif image_type in ["Screenshot / UI / digital interface", "UI / Screenshot", "Digital Interface"]:
            ui_str = ", ".join(ui_elements) if ui_elements else obj_str
            prompt_parts = [
                f"A sleek modern {style.lower()} digital web application interface dashboard",
                f"featuring {ui_str}" if ui_str else f"featuring {subj_str}",
                f"with visible interface elements [{text_str}]" if text_str else "",
                f"structured layout with {spatial_str}" if spatial_str else "",
                f"rendered in a sophisticated color scheme of {dom_colors_str}" if dom_colors_str else "",
                "modern Figma UI/UX kit aesthetics, crisp iconography, pixel-perfect alignment, high contrast typography, 8k UI mockup"
            ]
            main_prompt = ". ".join(p for p in prompt_parts if p).strip() + "."
            negative_prompt = "photograph, real world photo, 50mm, bokeh, dirty, noisy, outdoors, nature scene, camera grain, blurry"
            mj_prompt = f"/imagine prompt: {main_prompt} --v 6.0 --style raw --ar 16:9"
            sd_prompt = f"{main_prompt}, UI kit, Dribbble trending, Figma UI design, clean vector components, 8k resolution"

        # Handle Document / Text-Heavy
        elif image_type in ["Document / text-heavy image", "Document"]:
            doc_str = ", ".join(doc_elements) if doc_elements else obj_str
            prompt_parts = [
                f"A clean editorial document page layout with {doc_str}",
                f"displaying legible text elements [{text_str}]" if text_str else "",
                f"structured typographic hierarchy in {dom_colors_str}" if dom_colors_str else "",
                f"{environment}" if environment else "clean white paper background",
                "crisp margins, elegant modern typography, vector document graphic, editorial PDF layout, high contrast"
            ]
            main_prompt = ". ".join(p for p in prompt_parts if p).strip() + "."
            negative_prompt = "photorealistic photo, natural outdoors, 50mm lens, bokeh, blurry, organic landscape, camera noise"
            mj_prompt = f"/imagine prompt: {main_prompt} --v 6.0 --style raw --ar 3:4"
            sd_prompt = f"{main_prompt}, high-resolution vector PDF layout, sharp typography, minimalist graphic"

        # Handle Person / Portrait / Photograph / Artwork / Natural Scenes
        else:
            # Mode overrides
            effective_style = custom_style if (mode == "Creative Prompt" and custom_style != "Auto / Original") else style
            effective_lighting = custom_lighting if (mode == "Creative Prompt" and custom_lighting != "Auto / Original") else lighting
            effective_camera = custom_camera if (mode == "Creative Prompt" and custom_camera != "Auto / Original") else (f"{cam_shot} {cam_view} {cam_angle}".strip() or "Standard natural perspective")

            prompt_parts = [
                f"A high-fidelity {effective_style.lower()} image capturing {subj_str}" if subj_str else f"A high-fidelity {effective_style.lower()} image",
                f"attire and styling: {clothing_str}" if clothing_str else "",
                f"{action_str}" if action_str else "",
                f"interacting with {obj_str}" if (obj_str and obj_str != subj_str) else "",
                f"set within {environment}" if environment else "",
                f"spatial positioning: {spatial_str}" if spatial_str else "",
                f"illuminated by {effective_lighting}" if effective_lighting else "",
                f"framed from a {effective_camera}" if effective_camera else "",
                f"color palette featuring {dom_colors_str} ({color_temp} tones)" if dom_colors_str else "",
                "ultra-fine textures, authentic surface details, masterwork aesthetic clarity, sharp focus, 8k resolution"
            ]
            main_prompt = ". ".join(p for p in prompt_parts if p).strip() + "."
            negative_prompt = "blurry, low quality, deformed, distorted, extra limbs, bad anatomy, bad hands, missing fingers, artifacts, noisy texture, overexposed, underexposed"
            mj_prompt = f"/imagine prompt: {main_prompt} --v 6.0 --style raw --ar 16:9"
            sd_prompt = f"{main_prompt}, masterpiece, 8k resolution, highly detailed, sharp focus"

        attributes_display = {
            "Domain": image_type,
            "Confidence": f"{int(confidence * 100)}%",
            "Visual Style": style,
            "Primary Subject(s)": subj_str or "N/A",
            "Detected Objects": obj_str or "N/A",
            "Environment / Setting": environment or "N/A",
            "Actions & Poses": action_str or "N/A",
            "Spatial Layout": spatial_str or "N/A",
            "Clothing & Attire": clothing_str or "N/A",
            "Color Palette": f"{dom_colors_str} ({color_temp})" if dom_colors_str else "N/A",
            "Lighting": lighting or "N/A",
            "Camera & Perspective": f"{cam_shot} | {cam_view} | {cam_dof}" if (cam_shot or cam_view) else "Standard",
            "Text / OCR Content": text_str or "None detected",
        }

        return {
            "image_type": image_type,
            "confidence": confidence,
            "detailed_description": overall_desc or main_prompt,
            "master_prompt": main_prompt,
            "midjourney_prompt": mj_prompt,
            "stable_diffusion_prompt": sd_prompt,
            "dalle_prompt": main_prompt,
            "negative_prompt": negative_prompt,
            "attributes": attributes_display,
            "raw_vision_json": vision_json,
            "detected_objects": objects,
            "detected_text": text_content,
            "mode_used": mode,
            "engine_used": "Multimodal Vision Intelligence"
        }

    @classmethod
    def generate_prompt(
        cls,
        base_caption: str,
        vision_data: Optional[Dict[str, Any]] = None,
        style: str = "Photorealistic",
        lighting: str = "Auto / Natural",
        camera: str = "Auto / Standard",
        composition: str = "Auto / Balanced",
        mood: str = "Auto / Neutral",
        color_palette: str = "Auto / Natural",
        detail_level: str = "High Detail",
        prompt_length: str = "Balanced",
        engine_mode: str = "Template Engine",
        creativity: float = 0.7,
        llm_model: Optional[str] = None,
        llm_endpoint: str = "http://127.0.0.1:11434",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate grounded prompts routed by detected image type (Legacy/Offline fallback).
        """
        if seed is not None:
            random.seed(seed)

        image_type = vision_data.get("image_type", "Photograph / natural scene") if vision_data else "Photograph / natural scene"

        # Domain 1: Chart / Graph / Visualization
        if image_type == "Chart / graph / visualization":
            res = cls._build_chart_prompt(vision_data or {}, prompt_length)
            res.update({
                "base_caption": base_caption,
                "factual_description": vision_data.get("factual_description", base_caption) if vision_data else base_caption,
                "image_type": image_type,
                "engine_used": "Domain-Specific Chart Engine",
                "fallback_used": False,
                "fallback_reason": "",
            })
            return res

        # Domain 2: Document / Text-Heavy
        elif image_type == "Document / text-heavy image":
            res = cls._build_document_prompt(vision_data or {}, prompt_length)
            res.update({
                "base_caption": base_caption,
                "factual_description": vision_data.get("factual_description", base_caption) if vision_data else base_caption,
                "image_type": image_type,
                "engine_used": "Domain-Specific Document Engine",
                "fallback_used": False,
                "fallback_reason": "",
            })
            return res

        # Domain 3: Screenshot / UI
        elif image_type == "Screenshot / UI / digital interface":
            res = cls._build_ui_prompt(vision_data or {}, prompt_length)
            res.update({
                "base_caption": base_caption,
                "factual_description": vision_data.get("factual_description", base_caption) if vision_data else base_caption,
                "image_type": image_type,
                "engine_used": "Domain-Specific UI Engine",
                "fallback_used": False,
                "fallback_reason": "",
            })
            return res

        # Domain 4: Photographic & Natural Scenes (Person, Animal, Product, Landscape, Photo, Architecture)
        semantics = cls.parse_semantics(base_caption)

        if image_type == "Person / portrait" and vision_data:
            attire_text = vision_data.get("attire", "")
            if attire_text and attire_text != "Casual wear":
                semantics["subject"] = f"a person wearing stylish {attire_text} with natural portrait presence"
                semantics["objects"] = f"authentic {attire_text} and personal attire"
            elif not base_caption or base_caption in ["a subject in an environmental scene", ""]:
                semantics["subject"] = "a compelling person captured with natural portrait expression"

        # Preset strings
        style_info = STYLE_PRESETS.get(style, STYLE_PRESETS["Photorealistic"])
        style_desc = style_info["descriptor"]
        style_tags = style_info["engine_tags"]
        negative_prompt = style_info["negative"]

        lighting_desc = LIGHTING_PRESETS.get(lighting, LIGHTING_PRESETS["Auto / Natural"])
        camera_desc = CAMERA_PRESETS.get(camera, CAMERA_PRESETS["Auto / Standard"])
        composition_desc = COMPOSITION_PRESETS.get(composition, COMPOSITION_PRESETS["Auto / Balanced"])
        mood_desc = MOOD_PRESETS.get(mood, MOOD_PRESETS["Auto / Neutral"])
        color_desc = COLOR_PRESETS.get(color_palette, COLOR_PRESETS["Auto / Natural"])
        detail_desc = DETAIL_PRESETS.get(detail_level, DETAIL_PRESETS["High Detail"])

        structured_attributes = {
            "Domain": image_type,
            "Subject": semantics["subject"],
            "Environment": semantics["environment"],
            "Objects": semantics["objects"],
            "Action / Pose": semantics["action"],
            "Visual Style": f"{style} ({style_desc})",
            "Lighting": lighting_desc,
            "Mood": mood_desc,
            "Color Palette": color_desc,
            "Composition": composition_desc,
            "Camera Perspective": camera_desc,
            "Level of Detail": detail_desc,
        }

        if prompt_length == "Compact":
            template_main_prompt = (
                f"{style_desc} depicting {semantics['subject']} {semantics['action']} in {semantics['environment']}, "
                f"{lighting_desc}, {camera_desc}, {detail_desc}."
            )
        elif prompt_length == "Extended":
            template_main_prompt = (
                f"{style_desc.capitalize()}. A masterwork visual featuring {semantics['subject']} {semantics['action']}. "
                f"The scene is set within {semantics['environment']}, accompanied by {semantics['objects']}. "
                f"Illuminated by {lighting_desc}, evoking a {mood_desc}. "
                f"Composed with {composition_desc} from a {camera_desc}. "
                f"Finished in {color_desc}, rendering {detail_desc}, ultra-fine textures, and impeccable aesthetic clarity."
            )
        else:  # Balanced (Default)
            template_main_prompt = (
                f"{style_desc.capitalize()} featuring {semantics['subject']} {semantics['action']} in {semantics['environment']}. "
                f"{semantics['objects'].capitalize()}, enhanced by {lighting_desc} with a {mood_desc}. "
                f"Framed with {composition_desc}, {camera_desc}, {color_desc}, {detail_desc}."
            )

        final_prompt = template_main_prompt
        engine_used = "Template Engine"
        fallback_used = False
        fallback_reason = ""

        # Local LLM if requested
        if engine_mode in ["Local AI", "Local AI (Ollama)"]:
            llm_result = LocalLLMService.generate_enhanced_prompt(
                base_caption=base_caption,
                attributes=structured_attributes,
                model_name=llm_model or "llama3",
                creativity=creativity,
                prompt_length=prompt_length,
                endpoint=llm_endpoint,
            )

            if llm_result and isinstance(llm_result, dict) and llm_result.get("prompt"):
                final_prompt = llm_result.get("prompt", template_main_prompt)
                if llm_result.get("negative_prompt"):
                    negative_prompt = llm_result.get("negative_prompt")

                for key_attr, json_key in [
                    ("Subject", "subject"),
                    ("Environment", "environment"),
                    ("Visual Style", "style"),
                    ("Lighting", "lighting"),
                    ("Mood", "mood"),
                    ("Composition", "composition"),
                    ("Camera Perspective", "camera"),
                    ("Color Palette", "color_palette"),
                    ("Level of Detail", "detail_level"),
                ]:
                    if llm_result.get(json_key):
                        structured_attributes[key_attr] = llm_result[json_key]

                engine_used = f"Local AI ({llm_model or 'Ollama'})"
                fallback_used = False
            else:
                engine_used = "Template Engine (Fallback)"
                fallback_used = True
                fallback_reason = "Ollama service was not reachable or model did not respond in time."

        midjourney_prompt = f"/imagine prompt: {final_prompt}, {style_tags} --v 6.0 --style raw --ar 16:9"
        stable_diffusion_prompt = f"{final_prompt}, {style_tags}, {detail_desc}, highly detailed, sharp focus"
        dalle_prompt = final_prompt

        return {
            "base_caption": base_caption,
            "factual_description": vision_data.get("factual_description", base_caption) if vision_data else base_caption,
            "image_type": image_type,
            "main_prompt": final_prompt,
            "structured_attributes": structured_attributes,
            "midjourney_prompt": midjourney_prompt,
            "stable_diffusion_prompt": stable_diffusion_prompt,
            "dalle_prompt": dalle_prompt,
            "negative_prompt": negative_prompt,
            "engine_mode": engine_mode,
            "engine_used": engine_used,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }

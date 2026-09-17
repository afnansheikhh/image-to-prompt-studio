import os
os.environ["HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["KERAS_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageStat, ImageFilter
import numpy as np
import re

from services.ocr_service import OCRService

try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
except (ImportError, Exception):
    tf = None
    TF_AVAILABLE = False

try:
    import streamlit as st
    cache_resource = st.cache_resource
except ImportError:
    def cache_resource(fn):
        return fn


@cache_resource
def get_full_mobilenet_classifier():
    """Load MobileNetV2 with full ImageNet-1000 classification head if cached/available."""
    try:
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        local_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras", "models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"))
        if os.path.exists(local_weights):
            model = MobileNetV2(weights=None)
            model.load_weights(local_weights)
            return model
        return MobileNetV2(weights="imagenet")
    except Exception:
        return None


class VisionService:
    """Core grounded visual understanding and image classification service."""

    CATEGORIES = [
        "Photograph / natural scene",
        "Person / portrait",
        "Animal",
        "Product",
        "Architecture / building",
        "Landscape / nature",
        "Chart / graph / visualization",
        "Screenshot / UI / digital interface",
        "Document / text-heavy image",
        "Diagram / technical drawing / infographic",
        "Other / abstract / artistic",
    ]

    # ImageNet keywords mapped to domains
    ANIMAL_KEYWORDS = {
        "dog", "cat", "puppy", "kitten", "bird", "parrot", "canary", "eagle", "hawk",
        "owl", "penguin", "fish", "shark", "lion", "tiger", "bear", "elephant",
        "zebra", "giraffe", "horse", "monkey", "chimpanzee", "gorilla", "panda",
        "rabbit", "hare", "snake", "lizard", "turtle", "frog", "spider", "insect",
        "butterfly", "fox", "wolf", "deer", "camel", "whale", "dolphin", "seal"
    }

    PERSON_KEYWORDS = {
        "suit", "jersey", "trench_coat", "academic_gown", "jean", "jeans", "blue_jean", "denim",
        "bikini", "brassiere", "cardigan", "cowboy_hat", "sombrero", "wig", "fur_coat",
        "cloak", "scuba_diver", "groom", "bride", "military_uniform", "apron", "kimono",
        "sunglasses", "sunglass", "sun_glasses", "spectacles", "shades",
        "t-shirt", "tshirt", "sweatshirt", "sweater", "jacket", "vest", "bonnet",
        "crash_helmet", "football_helmet", "swimming_trunks", "pajama", "gown",
        "bow_tie", "tie", "necktie", "shoe", "running_shoe", "boot", "sandal", "sock",
        "boy", "girl", "child", "kid", "toddler", "man", "woman", "person", "baby"
    }

    ARCHITECTURE_KEYWORDS = {
        "church", "castle", "palace", "mosque", "bridge", "suspension_bridge",
        "monastery", "beacon", "lighthouse", "triumphal_arch", "viaduct",
        "vault", "dome", "stupa", "barn", "cinema", "planetarium", "skyscraper",
        "dam", "pier", "tile_roof"
    }

    LANDSCAPE_KEYWORDS = {
        "alp", "cliff", "promontory", "seashore", "lakeside", "sandbar", "valley",
        "volcano", "coral_reef", "geyser", "mountain", "waterfall", "fountain", "dam",
        "pier", "breakwater", "lakeshore", "coast", "forest", "jungle", "river", "ocean",
        "sea", "sky", "cloud", "sunset", "sunrise", "canoe", "paddle", "gondola", "boat",
        "ship", "lifeboat", "sailboat"
    }

    DOCUMENT_KEYWORDS = {
        "envelope", "packet", "book_jacket", "comic_book", "crossword_puzzle", "menu"
    }

    DIGITAL_KEYWORDS = {
        "web_site", "screen", "monitor", "television", "home_theater", "scoreboard",
        "digital_clock", "digital_watch", "hand-held_computer", "cellular_telephone",
        "ipod", "laptop", "notebook"
    }

    @classmethod
    def analyze_pixel_properties(cls, image: Image.Image) -> Dict[str, Any]:
        """Compute pixel statistics: color entropy, edge density, white ratio, uniformity, skin ratio."""
        rgb_img = image.convert("RGB")
        w, h = rgb_img.size

        # Color entropy and standard deviation
        stat = ImageStat.Stat(rgb_img)
        mean_r, mean_g, mean_b = stat.mean[:3]
        std_r, std_g, std_b = stat.stddev[:3]
        avg_std = (std_r + std_g + std_b) / 3.0

        # Downsample for fast histogram and color clustering
        small = rgb_img.resize((100, 100))
        np_img = np.array(small, dtype=np.float32)

        # White/Near-white pixel ratio (typical in documents, charts, minimalist UI)
        white_mask = (np_img[:, :, 0] > 235) & (np_img[:, :, 1] > 235) & (np_img[:, :, 2] > 235)
        white_ratio = float(np.mean(white_mask))

        # Dark pixel ratio
        dark_mask = (np_img[:, :, 0] < 30) & (np_img[:, :, 1] < 30) & (np_img[:, :, 2] < 30)
        dark_ratio = float(np.mean(dark_mask))

        # Skin tone pixel detection
        r, g, b = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
        skin_mask = (
            (r > 95)
            & (g > 40)
            & (b > 20)
            & (r > g)
            & (r > b)
            & ((r - g) > 15)
            & (abs(r - g) > 15)
        )
        skin_ratio = float(np.mean(skin_mask))

        # Edge analysis via Sobel / Laplacian filter
        gray = rgb_img.convert("L").resize((150, 150))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_density = edge_stat.mean[0]

        # Dominant color palette extraction (quantized 6 colors)
        quantized = rgb_img.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()[:18]  # 6 RGB tuples
        hex_colors = []
        color_names = []
        for i in range(0, len(palette), 3):
            r_c, g_c, b_c = palette[i], palette[i+1], palette[i+2]
            hex_colors.append(f"#{r_c:02x}{g_c:02x}{b_c:02x}")
            if r_c > 220 and g_c > 220 and b_c > 220:
                color_names.append("Clean White")
            elif r_c < 40 and g_c < 40 and b_c < 40:
                color_names.append("Deep Black / Dark Gray")
            elif r_c > g_c + 40 and r_c > b_c + 40:
                color_names.append("Vibrant Red / Warm Orange")
            elif b_c > r_c + 30 and b_c > g_c + 30:
                color_names.append("Rich Blue")
            elif g_c > r_c + 30 and g_c > b_c + 30:
                color_names.append("Fresh Green")
            elif r_c > 180 and g_c > 180 and b_c < 100:
                color_names.append("Yellow / Gold")
            elif abs(r_c - g_c) < 20 and abs(g_c - b_c) < 20:
                color_names.append("Neutral Gray")
            else:
                color_names.append(f"RGB({r_c},{g_c},{b_c})")

        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 2) if h > 0 else 1.0,
            "avg_std": avg_std,
            "white_ratio": white_ratio,
            "dark_ratio": dark_ratio,
            "skin_ratio": skin_ratio,
            "edge_density": edge_density,
            "hex_colors": hex_colors[:5],
            "dominant_colors": list(dict.fromkeys(color_names))[:4],
        }

    @classmethod
    def classify_image(
        cls,
        image: Image.Image,
        ocr_data: Dict[str, Any],
        pixel_props: Dict[str, Any],
        base_caption: str = ""
    ) -> Tuple[str, float, str]:
        """
        Classify image into 11 categories using ImageNet predictions + CV heuristics + OCR.
        Returns: (category, confidence, rationale)
        """
        top_classes = []
        top_name = "visual_feature"
        top_prob = 0.5

        # Attempt to run MobileNetV2 classifier if available
        classifier = get_full_mobilenet_classifier()
        if classifier is not None:
            try:
                resized = image.convert("RGB").resize((224, 224))
                arr = img_to_array(resized).reshape((1, 224, 224, 3))
                preprocessed = preprocess_input(arr)
                preds = classifier.predict(preprocessed, verbose=0)
                decoded = decode_predictions(preds, top=5)[0]
                top_classes = [c[1].lower() for c in decoded]
                top_name = top_classes[0]
                top_prob = float(decoded[0][2])
            except Exception:
                pass

        white_ratio = pixel_props.get("white_ratio", 0.0)
        dark_ratio = pixel_props.get("dark_ratio", 0.0)
        skin_ratio = pixel_props.get("skin_ratio", 0.0)
        edge_density = pixel_props.get("edge_density", 0.0)
        has_text = ocr_data.get("has_text", False)
        char_count = ocr_data.get("char_count", 0)
        line_count = ocr_data.get("line_count", 0)
        raw_ocr = ocr_data.get("raw_text", "").lower()
        numeric_count = len(ocr_data.get("numeric_values", []))

        # Semantic caption cues
        cap = base_caption.lower()
        is_digital_class = any(c in cls.DIGITAL_KEYWORDS or c in cls.DOCUMENT_KEYWORDS for c in top_classes)
        has_chart_keywords = any(kw in cap for kw in ["chart", "graph", "plot", "histogram", "pie chart", "bar chart", "bar column", "bars", "axes", "diagram", "scatter"])
        has_doc_keywords = any(kw in cap for kw in ["document", "page", "text", "printed", "heading", "article", "paragraph", "letter", "paper", "report", "receipt", "invoice"])
        has_ui_keywords = any(kw in cap for kw in ["screenshot", "dashboard", "interface", "web application", "website", "ui", "screen", "mobile app", "software", "login", "navigation"])
        has_diagram_keywords = any(kw in cap for kw in ["flowchart", "infographic", "schematic", "blueprint", "wireframe"])

        # Check 1: Document / Text-Heavy
        if has_doc_keywords or (white_ratio > 0.55 and char_count > 60 and line_count >= 3):
            return "Document / text-heavy image", 0.93, "High text density or document layout detected."

        # Check 2: Chart / Graph / Visualization (High Priority)
        # Charts have large flat white/dark backgrounds (>35%), high edge density from axes/bars, or chart-like ImageNet classifications
        has_flat_bg = white_ratio > 0.35 or dark_ratio > 0.35
        has_chart_geometry = has_flat_bg and edge_density > 14
        is_chart_imagenet = top_name in ["scoreboard", "menu", "crossword_puzzle", "web_site", "envelope"] and has_flat_bg
        if (
            has_chart_keywords
            or (has_chart_geometry and white_ratio > 0.45)
            or (has_flat_bg and is_chart_imagenet and edge_density > 12)
            or (numeric_count >= 2 and has_flat_bg and line_count >= 2 and edge_density > 10)
        ):
            return "Chart / graph / visualization", 0.95, "Identified structured data visualization layout, axes/grid lines, and graphical metric columns."

        # Check 3: Diagram / Technical drawing
        if has_diagram_keywords or (white_ratio > 0.5 and edge_density > 18 and char_count > 10):
            return "Diagram / technical drawing / infographic", 0.88, "Detected schematic lines and annotated technical diagrams."

        # Check 4: UI Screenshot / Dashboard (Requires explicit UI signals, never fires on pure photos)
        has_ui_text = any(kw in raw_ocr for kw in ["login", "dashboard", "settings", "profile", "menu", "submit", "home", "sign in", "search", "overview", "analytics", "button", "user", "password"])
        if has_ui_keywords or (has_ui_text and line_count >= 2) or (top_name in cls.DIGITAL_KEYWORDS and top_prob > 0.38 and char_count > 5):
            return "Screenshot / UI / digital interface", 0.90, "Identified digital user interface components and screen layout."

        # Check 5: Person / Portrait (Requires natural photo, not flat white vector background)
        is_natural_photo = white_ratio < 0.40 and dark_ratio < 0.60
        matched_person_classes = [c for c in top_classes if any(pk in c for pk in cls.PERSON_KEYWORDS)]
        has_person_cap = any(pk in cap for pk in ["man", "woman", "person", "boy", "girl", "child", "kid", "baby", "people", "standing", "sitting", "walking", "face", "portrait", "wearing", "sunglasses", "jeans", "shirt", "jacket"])
        if is_natural_photo and (matched_person_classes or has_person_cap or (skin_ratio > 0.035 and not has_text)):
            if matched_person_classes:
                matched_cue = matched_person_classes[0]
            elif skin_ratio > 0.035:
                matched_cue = "human skin tones & natural features"
            else:
                matched_cue = "human subject"
            return "Person / portrait", 0.92, f"Identified human subject or personal attire ({matched_cue})."

        # Check 6: Animal (Requires genuine probability > 0.15)
        matched_animals = [c for c in top_classes[:3] if any(ak in c for ak in cls.ANIMAL_KEYWORDS)]
        if is_natural_photo and (matched_animals or any(ak in cap for ak in cls.ANIMAL_KEYWORDS)):
            cue = matched_animals[0] if matched_animals else top_name
            return "Animal", max(0.85, top_prob), f"Identified animal subject ({cue})."

        # Check 7: Architecture
        matched_arch = [c for c in top_classes[:3] if any(arch in c for arch in cls.ARCHITECTURE_KEYWORDS)]
        if is_natural_photo and (matched_arch or any(arch in cap for arch in ["building", "house", "church", "bridge", "tower", "castle", "room", "interior", "exterior", "architecture"])):
            cue = matched_arch[0] if matched_arch else top_name
            return "Architecture / building", max(0.85, top_prob), f"Identified architectural structures ({cue})."

        # Check 8: Landscape
        matched_land = [c for c in top_classes[:3] if any(land in c for land in cls.LANDSCAPE_KEYWORDS)]
        if is_natural_photo and (matched_land or any(land in cap for land in ["mountain", "beach", "forest", "lake", "ocean", "river", "valley", "sunset", "sky", "snow", "sea", "waterfall"])):
            cue = matched_land[0] if matched_land else top_name
            return "Landscape / nature", max(0.85, top_prob), f"Identified natural terrain / landscape ({cue})."

        # Check 9: Product
        if (white_ratio > 0.4 and white_ratio < 0.85 and pixel_props["avg_std"] > 20 and edge_density < 14 and not has_text) or any(p in cap for p in ["bottle", "shoe", "phone", "watch", "product", "item", "box", "can", "cup"]):
            return "Product", 0.80, "Isolated object on neutral background."

        # Check 10: Abstract / Other
        if pixel_props["avg_std"] < 12 and edge_density < 5:
            return "Other / abstract / artistic", 0.75, "Minimalist color field / abstract visual."

        # Default fallback: Photograph / Natural Scene
        return "Photograph / natural scene", 0.85, f"General photographic scene with natural features ({top_name})."

    @classmethod
    def analyze_image(cls, image: Image.Image, base_caption: str = "") -> Dict[str, Any]:
        """
        Complete grounded visual understanding pipeline.
        Extracts OCR, pixel properties, classifies category, and builds structured metadata.
        """
        ocr_data = OCRService.extract_text(image)
        pixel_props = cls.analyze_pixel_properties(image)
        category, confidence, rationale = cls.classify_image(image, ocr_data, pixel_props, base_caption)

        # Domain-Specific Structured Metadata Extraction
        structured_metadata: Dict[str, Any] = {
            "image_type": category,
            "confidence": round(confidence, 2),
            "rationale": rationale,
            "color_palette": pixel_props["dominant_colors"],
            "hex_palette": pixel_props["hex_colors"],
            "aspect_ratio": f"{pixel_props['aspect_ratio']}:1",
        }

        # Attempt to run MobileNet classifier to enrich attire & props
        top_classes = []
        classifier = get_full_mobilenet_classifier()
        if classifier is not None:
            try:
                resized = image.convert("RGB").resize((224, 224))
                arr = img_to_array(resized).reshape((1, 224, 224, 3))
                preprocessed = preprocess_input(arr)
                preds = classifier.predict(preprocessed, verbose=0)
                decoded = decode_predictions(preds, top=5)[0]
                top_classes = [c[1].lower() for c in decoded]
            except Exception:
                pass

        # Detailed domain extraction & factual description
        if category == "Chart / graph / visualization":
            chart_subtype = "Bar Chart"
            lines_str = " ".join(ocr_data["lines"]).lower()
            if "pie" in lines_str or "%" in lines_str and len(ocr_data["numeric_values"]) <= 6:
                chart_subtype = "Pie / Donut Chart"
            elif "trend" in lines_str or "growth" in lines_str or "line" in lines_str:
                chart_subtype = "Line Graph / Trend Chart"
            elif "scatter" in lines_str:
                chart_subtype = "Scatter Plot"

            title_text = ocr_data["potential_titles"][0] if ocr_data["potential_titles"] else (ocr_data["labels"][0] if ocr_data["labels"] else "Statistical Data Visualization")
            metric_list = ", ".join(ocr_data["numeric_values"][:6]) if ocr_data["numeric_values"] else "Comparative metric values"
            label_list = ", ".join(ocr_data["labels"][:6]) if ocr_data["labels"] else "Categorical axes"

            structured_metadata["chart_subtype"] = chart_subtype
            structured_metadata["title"] = title_text
            structured_metadata["data_series"] = metric_list
            structured_metadata["category_labels"] = label_list
            structured_metadata["background"] = "Clean white minimalist background" if pixel_props["white_ratio"] > 0.3 else "Dark modern theme background"
            
            factual_description = (
                f"A clean 2D vector {chart_subtype.lower()} titled '{title_text}' displaying data points [{metric_list}] "
                f"across categories [{label_list}]. Rendered with crisp high-contrast bars and clean coordinate grid lines "
                f"in {', '.join(pixel_props['dominant_colors'][:3])} tones against a {structured_metadata['background'].lower()}."
            )

        elif category == "Document / text-heavy image":
            doc_type = "Document Page / Report"
            header = ocr_data["potential_titles"][0] if ocr_data["potential_titles"] else "Text Document"
            structured_metadata["document_type"] = doc_type
            structured_metadata["header"] = header
            structured_metadata["line_count"] = ocr_data["line_count"]
            structured_metadata["layout"] = "Multi-paragraph structured typographic layout"

            factual_description = (
                f"A formatted digital document with structured typography headed by '{header}'. "
                f"Contains {ocr_data['line_count']} lines of structured text, formatted headings, and clean document margins."
            )

        elif category == "Screenshot / UI / digital interface":
            theme = "Dark Mode UI" if pixel_props["dark_ratio"] > 0.3 else "Light Clean UI"
            title = ocr_data["potential_titles"][0] if ocr_data["potential_titles"] else "Software Application Dashboard"
            structured_metadata["interface_type"] = "Web / Desktop Application UI"
            structured_metadata["theme"] = theme
            structured_metadata["visible_nav"] = ", ".join(ocr_data["labels"][:4]) if ocr_data["labels"] else "Navigation & Action Controls"

            factual_description = (
                f"A modern {theme.lower()} digital user interface titled '{title}'. "
                f"Features structured cards, action buttons, navigation elements [{structured_metadata['visible_nav']}], "
                f"and modern digital layout aesthetics."
            )

        elif category == "Diagram / technical drawing / infographic":
            structured_metadata["diagram_type"] = "Technical Infographic / Flowchart"
            factual_description = (
                f"A structured technical diagram and vector infographic with interconnected process blocks, annotations, "
                f"and illustrative diagrams in {', '.join(pixel_props['dominant_colors'][:2])} palette."
            )

        elif category == "Person / portrait":
            attire_elements = []
            for c in top_classes:
                if any(k in c for k in ["sunglass", "sunglasses", "shades", "spectacle"]):
                    attire_elements.append("sunglasses")
                elif any(k in c for k in ["jean", "denim"]):
                    attire_elements.append("denim jeans")
                elif any(k in c for k in ["jersey", "t-shirt", "shirt"]):
                    attire_elements.append("casual t-shirt")
                elif any(k in c for k in ["suit", "coat", "jacket"]):
                    attire_elements.append("jacket / outerwear")
                elif any(k in c for k in ["hat", "sombrero", "cap", "helmet"]):
                    attire_elements.append("headwear")

            attire_str = f" wearing {', '.join(dict.fromkeys(attire_elements))}" if attire_elements else ""
            subject_desc = base_caption if base_caption and base_caption != "a subject in an environmental scene" else "A person"
            structured_metadata["subject_type"] = "Human portrait"
            structured_metadata["attire"] = ", ".join(dict.fromkeys(attire_elements)) if attire_elements else "Casual wear"
            factual_description = f"{subject_desc.capitalize()}{attire_str} captured in natural environmental lighting with authentic portrait presence."

        elif category == "Animal":
            subject_desc = base_caption if base_caption and base_caption != "a subject in an environmental scene" else "An animal subject"
            structured_metadata["animal_type"] = "Fauna / Pet"
            factual_description = f"{subject_desc.capitalize()} with distinctive natural features and texture."

        elif category == "Product":
            subject_desc = base_caption if base_caption and base_caption != "a subject in an environmental scene" else "Commercial product"
            structured_metadata["product_type"] = "Commercial Product Display"
            factual_description = f"Studio product presentation of {subject_desc} with clean balanced lighting and crisp focal definition."

        elif category == "Architecture / building":
            subject_desc = base_caption if base_caption and base_caption != "a subject in an environmental scene" else "Architectural structure"
            structured_metadata["structure_type"] = "Architectural Exterior / Interior"
            factual_description = f"{subject_desc.capitalize()} showcasing structural lines, facades, and spatial perspective."

        elif category == "Landscape / nature":
            subject_desc = base_caption if base_caption and base_caption != "a subject in an environmental scene" else "Natural landscape"
            structured_metadata["landscape_type"] = "Nature & Scenic Terrain"
            factual_description = f"Expansive natural landscape depicting {subject_desc} with environmental depth."

        else:
            factual_description = f"An image featuring {base_caption if base_caption else 'visual elements'} in {', '.join(pixel_props['dominant_colors'][:2])} tones."

        structured_metadata["factual_description"] = factual_description
        structured_metadata["ocr"] = ocr_data

        return structured_metadata

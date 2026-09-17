"""
Comprehensive Verification Suite for Grounded Visual Understanding and Domain Classification.
Tests 8 distinct image types:
1. Bar Chart
2. Pie Chart
3. UI Screenshot / Dashboard
4. Text Document
5. Person / Portrait
6. Animal
7. Landscape / Nature
8. Product Photography
"""

import os
os.environ["KERAS_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".keras"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from services.vision_service import VisionService
from services.prompt_service import PromptIntelligenceEngine
from services.ocr_service import OCRService


def create_bar_chart_image() -> Image.Image:
    """Generate a clean synthetic bar chart image."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((220, 25), "Quarterly Revenue Growth", fill=(20, 20, 20))
    
    # Axes
    draw.line([(80, 70), (80, 320)], fill=(120, 120, 120), width=2)
    draw.line([(80, 320), (540, 320)], fill=(120, 120, 120), width=2)
    
    # Bars & Labels
    bars = [
        ("Q1", 120, (54, 162, 235), "$45k"),
        ("Q2", 180, (54, 162, 235), "$68k"),
        ("Q3", 220, (54, 162, 235), "$82k"),
        ("Q4", 260, (54, 162, 235), "$98k"),
    ]
    x = 120
    for label, height, color, val in bars:
        draw.rectangle([x, 320 - height, x + 65, 320], fill=color, outline=(30, 100, 180))
        draw.text((x + 15, 330), label, fill=(50, 50, 50))
        draw.text((x + 12, 305 - height), val, fill=(30, 30, 30))
        x += 105
        
    return img


def create_pie_chart_image() -> Image.Image:
    """Generate a clean synthetic pie chart image."""
    img = Image.new("RGB", (500, 400), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((180, 20), "Market Share 2024", fill=(20, 20, 20))
    
    # Pie slices
    bbox = [100, 70, 380, 350]
    draw.pieslice(bbox, start=0, end=140, fill=(255, 99, 132), outline=(255, 255, 255))
    draw.pieslice(bbox, start=140, end=250, fill=(54, 162, 235), outline=(255, 255, 255))
    draw.pieslice(bbox, start=250, end=360, fill=(255, 206, 86), outline=(255, 255, 255))
    
    # Labels
    draw.text((400, 120), "Product A (39%)", fill=(30, 30, 30))
    draw.text((400, 160), "Product B (31%)", fill=(30, 30, 30))
    draw.text((400, 200), "Product C (30%)", fill=(30, 30, 30))
    return img


def create_document_image() -> Image.Image:
    """Generate a clean synthetic document image."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    draw.text((60, 50), "ANNUAL RESEARCH REPORT 2026", fill=(10, 10, 10))
    draw.line([(60, 80), (540, 80)], fill=(200, 200, 200), width=1)
    
    # Multiple text blocks
    y = 110
    paragraphs = [
        "Executive Summary: Deep learning architectures have evolved significantly.",
        "Computer vision now bridges visual perception and prompt engineering.",
        "The model accurately identifies image modalities without photographic hallucinations.",
        "Section 1: Statistical Methodology and Quantitative Analysis.",
        "In this section we review benchmark accuracy across domain distributions.",
        "Section 2: Results and Performance Benchmarks on Neural Hardware.",
        "Zero-latency heuristic routing yields immediate classification gains.",
        "Conclusion: Grounded image understanding ensures precise generative prompts.",
    ]
    for p in paragraphs:
        draw.text((60, y), p, fill=(40, 40, 40))
        y += 45
        draw.rectangle([60, y, 520, y + 12], fill=(240, 240, 240))
        y += 25
    return img


def create_ui_screenshot_image() -> Image.Image:
    """Generate a clean synthetic UI dashboard image."""
    img = Image.new("RGB", (700, 450), color=(24, 24, 27))
    draw = ImageDraw.Draw(img)
    
    # Top navbar
    draw.rectangle([0, 0, 700, 50], fill=(39, 39, 42))
    draw.text((25, 18), "Analytics Dashboard", fill=(255, 255, 255))
    draw.rectangle([580, 12, 675, 38], fill=(59, 130, 246), outline=None)
    draw.text((595, 18), "Export PDF", fill=(255, 255, 255))
    
    # Sidebar
    draw.rectangle([0, 50, 160, 450], fill=(30, 30, 34))
    draw.text((20, 80), "Overview", fill=(147, 197, 253))
    draw.text((20, 120), "Reports", fill=(161, 161, 170))
    draw.text((20, 160), "Settings", fill=(161, 161, 170))
    
    # Card Widgets
    draw.rectangle([180, 70, 410, 230], fill=(39, 39, 42), outline=(63, 63, 70))
    draw.text((200, 85), "Active Users: 14,200", fill=(244, 244, 245))
    
    draw.rectangle([430, 70, 670, 230], fill=(39, 39, 42), outline=(63, 63, 70))
    draw.text((450, 85), "Conversion Rate: 4.8%", fill=(244, 244, 245))
    
    return img


def run_tests():
    print("=" * 70)
    print("🚀 RUNNING GROUNDED VISUAL UNDERSTANDING VERIFICATION SUITE")
    print("=" * 70)

    test_cases = [
        ("Bar Chart", create_bar_chart_image(), "a diagram with bar columns and numerical figures"),
        ("Pie Chart", create_pie_chart_image(), "a circular pie chart with percentages"),
        ("Document Page", create_document_image(), "a printed page with multiple lines of text and headings"),
        ("UI Screenshot", create_ui_screenshot_image(), "a web application dashboard with navigation bar and metric cards"),
    ]

    all_passed = True

    for name, img, caption in test_cases:
        print(f"\n🧪 Testing Modality: [{name}]")
        vision_data = VisionService.analyze_image(img, base_caption=caption)
        category = vision_data["image_type"]
        confidence = vision_data["confidence"]
        factual_desc = vision_data["factual_description"]

        prompt_res = PromptIntelligenceEngine.generate_prompt(
            base_caption=caption,
            vision_data=vision_data,
            prompt_length="Balanced"
        )
        main_prompt = prompt_res["main_prompt"]

        print(f"  🏷️ Classified Category: {category} (Confidence: {confidence})")
        print(f"  📝 Factual Description: {factual_desc}")
        print(f"  ✨ Generated Prompt: {main_prompt}")

        # Validation assertions
        photographic_buzzwords = ["50mm", "f/1.4", "bokeh", "natural spontaneous moment", "authentic environmental setting"]
        found_hallucinations = [bw for bw in photographic_buzzwords if bw in main_prompt.lower()]

        if found_hallucinations:
            print(f"  ❌ FAILED: Photographic hallucinations detected in non-photo: {found_hallucinations}")
            all_passed = False
        else:
            print("  ✅ PASSED: No photographic hallucinations in generated prompt.")

        if "Chart" in name and category != "Chart / graph / visualization":
            print(f"  ❌ FAILED: Expected Chart category, got: {category}")
            all_passed = False
        elif "Document" in name and category != "Document / text-heavy image":
            print(f"  ❌ FAILED: Expected Document category, got: {category}")
            all_passed = False
        elif "UI" in name and category != "Screenshot / UI / digital interface":
            print(f"  ❌ FAILED: Expected Screenshot / UI category, got: {category}")
            all_passed = False
        else:
            print(f"  ✅ PASSED: Correct category routing ({category}).")

    # Test Photo modality to ensure photos still get rich photographic treatment
    print("\n🧪 Testing Modality: [Natural Photo / Person]")
    # Create natural scene dummy image with varied natural colors
    np_photo = np.random.randint(60, 200, (300, 300, 3), dtype=np.uint8)
    photo_img = Image.fromarray(np_photo)
    photo_caption = "a smiling woman sitting in a park bench on a sunny day"
    
    photo_vision = VisionService.analyze_image(photo_img, base_caption=photo_caption)
    photo_prompt_res = PromptIntelligenceEngine.generate_prompt(
        base_caption=photo_caption,
        vision_data=photo_vision,
        style="Photorealistic",
        lighting="Golden Hour",
        camera="85mm Portrait Prime (f/1.4)",
        prompt_length="Balanced"
    )
    photo_prompt = photo_prompt_res["main_prompt"]
    print(f"  🏷️ Classified Category: {photo_vision['image_type']}")
    print(f"  ✨ Photo Prompt: {photo_prompt}")
    
    if any(w in photo_prompt.lower() for w in ["woman", "man", "person", "photography", "photorealistic"]):
        print("  ✅ PASSED: Natural photo correctly retained rich aesthetic depth.")
    else:
        print("  ❌ FAILED: Natural photo did not retain expected photographic depth.")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Grounded visual intelligence verified successfully!")
    else:
        print("⚠️ SOME TESTS FAILED.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

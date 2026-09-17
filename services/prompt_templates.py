"""
Prompt Templates & Visual Attribute Definitions for AI Image Prompt Generation.
"""

STYLE_PRESETS = {
    "Photorealistic": {
        "descriptor": "ultra-realistic award-winning photography, true-to-life textures, authentic imperfections, unedited RAW photo quality",
        "engine_tags": "photorealistic, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm Superia 400",
        "negative": "cartoon, 3d render, anime, illustration, painting, oversaturated, plastic skin, drawing, deformed",
    },
    "Cinematic": {
        "descriptor": "cinematic movie still from a high-budget blockbuster, dramatic atmosphere, anamorphic lens flare, Panavision 70mm, color graded",
        "engine_tags": "cinematic shot, 35mm film, dramatic lighting, depth of field, color graded, imax, cinematic composition",
        "negative": "flat lighting, home video, amateur snapshot, cartoon, blurry, low resolution, blown out highlights",
    },
    "Anime": {
        "descriptor": "stunning Japanese anime key visual, Studio Ghibli and Makoto Shinkai aesthetic, clean cel-shaded linework, expressive emotion, vibrant sky",
        "engine_tags": "anime aesthetic, Makoto Shinkai style, vibrant colors, detailed background, master anime illustration, pixiv trending",
        "negative": "photorealistic, 3d photograph, dull colors, western comic, blurry, deformed anatomy",
    },
    "Digital Art": {
        "descriptor": "breathtaking digital concept art, trending on ArtStation, dynamic brushwork, intricate detailing, rich atmospheric perspective",
        "engine_tags": "digital concept art, ArtStation trending, intricate details, epic lighting, matte painting, masterpiece",
        "negative": "photograph, pixelated, amateur, flat colors, low quality, unfinished",
    },
    "3D Render": {
        "descriptor": "photorealistic 3D CGI render, rendered in Unreal Engine 5 with Octane Render, Ray-traced global illumination, clean geometry, physically based materials",
        "engine_tags": "Unreal Engine 5, Octane Render, 8k render, raytracing, subsurface scattering, ambient occlusion",
        "negative": "2d sketch, drawing, low poly, noisy, grainy, flat",
    },
    "Fantasy": {
        "descriptor": "magical high-fantasy concept illustration, enchanting atmospheric particles, mythic aura, whimsical and otherworldly lighting",
        "engine_tags": "high fantasy, magical glow, epic concept art, intricate fantasy details, ethereal atmosphere",
        "negative": "modern technology, contemporary city, mundane, flat, lowres",
    },
    "Cyberpunk": {
        "descriptor": "futuristic cyberpunk scene, wet reflective asphalt, holographic neon signage, high-tech dystopian urbanism, synthwave glow",
        "engine_tags": "cyberpunk, neon glow, wet reflections, sci-fi city, volumetric fog, high tech dystopian",
        "negative": "pastoral, medieval, natural daylight, vintage 1800s, muted colors",
    },
    "Product Photography": {
        "descriptor": "commercial studio product photography, clean backdrop, crisp reflections, precision rim lighting, pristine commercial advertisement grade",
        "engine_tags": "commercial product shot, studio lighting, clean background, sharp focus, 8k, advertising quality",
        "negative": "messy, cluttered, dirty, bad lighting, out of focus, amateur",
    },
    "Fashion Photography": {
        "descriptor": "high-fashion editorial magazine cover shot, Vogue aesthetic, striking model styling, artistic studio lighting, haute couture elegance",
        "engine_tags": "high fashion editorial, Vogue magazine, couture styling, studio strobe, elegant composition, sharp model focus",
        "negative": "casual snapshot, poor posture, bad lighting, sloppy, low quality",
    },
    "Minimalist": {
        "descriptor": "minimalist architectural aesthetic, ample negative space, elegant simplicity, pure geometric lines, subtle tonal palette",
        "engine_tags": "minimalism, negative space, clean lines, serene simplicity, fine art composition",
        "negative": "cluttered, busy, chaotic, noisy, chaotic details, oversaturated",
    },
}

LIGHTING_PRESETS = {
    "Auto / Natural": "natural organic ambient lighting with balanced exposure",
    "Golden Hour": "warm golden hour sunlight, long soft shadows, amber sun flare, glowing rim light",
    "Dramatic Studio": "dramatic studio key light with deep Rembrandt shadows, sculpted high-contrast illumination",
    "Cyberpunk Neon": "vibrant dual-tone neon backlighting with pink and cyan glow, reflective highlights",
    "Soft Diffused": "gentle diffused overcast lighting, wrap-around soft light, zero harsh shadows",
    "Volumetric Sunbeams": "atmospheric god rays streaming through air, volumetric haze, cinematic light shafts",
    "Moody Low-Key": "dark moody low-key lighting, deep blacks, subtle chiaroscuro accents",
    "High-Key Crisp": "bright high-key illumination, immaculate clean whites, shadowless crystal clarity",
}

CAMERA_PRESETS = {
    "Auto / Standard": "eye-level perspective with standard 50mm natural focal length",
    "35mm Street Lens": "35mm documentary lens, candid street photography angle, organic environmental context",
    "85mm Portrait (f/1.4 Bokeh)": "85mm prime lens at f/1.4 aperture, creamy shallow depth-of-field bokeh, tack-sharp subject",
    "Ultra-wide 16mm": "ultra-wide 16mm focal length, expansive field of view, dynamic perspective",
    "Macro Close-Up": "extreme macro lens, razor-sharp micro-detail and intricate surface texture",
    "Drone Aerial": "dramatic high-angle drone aerial view, wide landscape scale, cinematic bird's-eye view",
    "Cinematic Anamorphic": "widescreen anamorphic 2.39:1 aspect ratio, horizontal blue lens streaks, cinematic framing",
}

COMPOSITION_PRESETS = {
    "Auto / Balanced": "harmonious visual balance with well-proportioned negative space",
    "Rule of Thirds": "classic rule of thirds alignment, offset focal point with dynamic visual weight",
    "Centered Symmetry": "striking centered symmetrical composition, perfect geometric alignment",
    "Dynamic Dutch Angle": "tilted Dutch angle, creating tension and kinetic motion",
    "Wide Establishing Scene": "grand wide establishing shot contextualizing the subject within the full environment",
    "Leading Lines & Depth": "strong foreground leading lines guiding the viewer into deep background layers",
}

MOOD_PRESETS = {
    "Auto / Neutral": "authentic, lifelike and engaging ambience",
    "Ethereal & Dreamy": "dreamy, otherworldly, serene and poetic aura",
    "Dark & Gritty": "gritty, suspenseful, mysterious and intense atmosphere",
    "Serene & Peaceful": "tranquil, calm, zen-like and peaceful atmosphere",
    "Vibrant & Energetic": "dynamic, lively, uplifting and exuberant mood",
    "Epic & Heroic": "grand, awe-inspiring, epic and monumental tone",
    "Nostalgic Vintage": "wistful, nostalgic, retro 70s-90s film warmth",
}

COLOR_PRESETS = {
    "Auto / Natural": "natural lifelike colors and true-to-life tones",
    "Vibrant & Saturated": "rich vibrant color saturation, punchy primary tones and lively contrast",
    "Warm Earthy & Amber": "warm amber, terracotta, ochre and rich earthy tones",
    "Cool Teal & Orange": "classic cinematic teal and orange color grading with deep contrast",
    "Neon & Electric": "electric magenta, ultraviolet, laser cyan and glowing neon palette",
    "Muted Monochrome": "subtle muted desaturated color palette with delicate tonal nuances",
    "Pastel & Soft": "soft pastel tones, lavender, peach, powder blue and creamy highlights",
}

DETAIL_PRESETS = {
    "Standard": "clean sharp details, realistic focus",
    "High Detail": "high-resolution micro-textures, intricate surface details, crisp edge definition",
    "Ultra Hyper-Detailed 8K": "hyper-detailed 8k resolution, subsurface scattering, tactile textures, masterwork fidelity",
}

import os
os.environ["HOME"] = os.path.abspath(os.path.dirname(__file__))
os.environ["KERAS_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".keras"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
import random
import time

from services.caption_service import generate_base_caption
from services.prompt_service import PromptIntelligenceEngine
from services.vision_api_service import VisionAPIService
from services.groq_service import GroqService
from services.config_service import ConfigService

# Page Configuration
st.set_page_config(
    page_title="PROMPT STUDIO // GROQ VISION",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load persistent config
saved_cfg = ConfigService.load_config()

# Initialize Session State
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "image_rotation" not in st.session_state:
    st.session_state.image_rotation = 0
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = saved_cfg.get("groq_api_key", "")
if "groq_vision_model" not in st.session_state:
    st.session_state.groq_vision_model = saved_cfg.get("groq_vision_model", "qwen/qwen3.8-27b")
if "groq_test_status" not in st.session_state:
    st.session_state.groq_test_status = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# Neo-Brutalism Medium Yellow & White Design System
st.markdown("""
<style>
    @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;800;900&family=Inter:wght@400;600;700;800&display=swap");
    
    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Space Grotesk', 'Inter', -apple-system, sans-serif !important;
        background-color: #FFFDF0 !important;
        color: #000000 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #FFFDF0 !important;
        border-right: 3.5px solid #000000 !important;
        box-shadow: 4px 0px 0px #000000 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Header */
    .neo-hero-container {
        text-align: center;
        margin-top: 0.2rem;
        margin-bottom: 1.8rem;
    }
    
    .neo-hero-badge {
        display: inline-block;
        background: #000000;
        color: #FFE600;
        font-weight: 900;
        font-size: 13px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 5px 16px;
        border: 3px solid #000000;
        box-shadow: 4px 4px 0px #000000;
        margin-bottom: 0.8rem;
        border-radius: 4px;
    }
    
    .neo-hero-title {
        font-size: 3.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #000000;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }
    
    .neo-hero-title span {
        background: #FFE600;
        padding: 2px 14px;
        border: 3.5px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        display: inline-block;
        border-radius: 6px;
    }
    
    .neo-hero-sub {
        font-size: 1.18rem;
        font-weight: 700;
        color: #222222;
        margin-top: 1rem;
        letter-spacing: -0.01em;
    }
    
    /* Neo-Brutalist Cards */
    .neo-card {
        background: #FFFFFF;
        border: 3.5px solid #000000;
        border-radius: 8px;
        padding: 1.4rem;
        box-shadow: 6px 6px 0px #000000;
        margin-bottom: 1.4rem;
    }
    
    .neo-section-label {
        font-size: 13px;
        font-weight: 900;
        color: #000000;
        background: #FFE600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 5px 14px;
        border: 3px solid #000000;
        box-shadow: 4px 4px 0px #000000;
        border-radius: 4px;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    .neo-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 800;
        background: #FFE600;
        color: #000000;
        border: 2.5px solid #000000;
        box-shadow: 3px 3px 0px #000000;
        margin: 0 8px 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    .neo-pill-white {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 800;
        background: #FFFFFF;
        color: #000000;
        border: 2.5px solid #000000;
        box-shadow: 3px 3px 0px #000000;
        margin: 0 8px 10px 0;
    }
    
    .neo-detail-text {
        font-size: 1.02rem;
        line-height: 1.65;
        font-weight: 600;
        color: #000000;
        background: #FFFFFF;
        border: 3.5px solid #000000;
        box-shadow: 5px 5px 0px #000000;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.3rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: #FFE600 !important;
        color: #000000 !important;
        border: 3.5px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: 900 !important;
        font-size: 14px !important;
        padding: 0.6rem 1.6rem !important;
        box-shadow: 4px 4px 0px #000000 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        transition: all 0.12s ease-in-out !important;
    }
    
    .stButton>button:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px #000000 !important;
        background: #FFF04D !important;
    }
    
    .stButton>button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 0px 0px 0px #000000 !important;
    }
    
    .stDownloadButton>button {
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 3.5px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: 900 !important;
        font-size: 14px !important;
        box-shadow: 4px 4px 0px #000000 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    
    .stDownloadButton>button:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px #000000 !important;
        background: #FFE600 !important;
    }
    
    /* File Uploader Clean Styling */
    [data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 3.5px dashed #000000 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        box-shadow: 5px 5px 0px #000000 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background: #FFFDF0 !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploaderDropzone"] div, [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button {
        background: #FFE600 !important;
        color: #000000 !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 3px 3px 0px #000000 !important;
        font-weight: 900 !important;
        border-radius: 6px !important;
    }
    
    /* Info Callout */
    [data-testid="stAlert"] {
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 3.5px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }
    
    /* Textareas and Inputs */
    textarea, input[type="text"], input[type="password"], [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-radius: 8px !important;
        font-family: 'Space Grotesk', monospace !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }
    
    textarea:focus, input:focus {
        border-color: #000000 !important;
        box-shadow: 6px 6px 0px #000000 !important;
    }
    
    /* Popover Settings Button */
    [data-testid="stPopover"]>button {
        background: #FFFFFF !important;
        border: 3px solid #000000 !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-radius: 8px !important;
        font-weight: 900 !important;
        font-size: 13px !important;
        color: #000000 !important;
        text-transform: uppercase !important;
        margin-top: 10px;
    }

    /* Popover Body & Content */
    [data-testid="stPopoverBody"] {
        background-color: #FFFFFF !important;
        border: 3.5px solid #000000 !important;
        box-shadow: 6px 6px 0px #000000 !important;
        border-radius: 10px !important;
        padding: 1.2rem !important;
        color: #000000 !important;
    }
    [data-testid="stPopoverBody"] * {
        color: #000000 !important;
    }
    [data-testid="stPopoverBody"] label {
        font-weight: 800 !important;
        text-transform: uppercase !important;
        font-size: 13px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF !important;
        border: 3px solid #000000 !important;
        box-shadow: 3px 3px 0px #000000 !important;
        border-radius: 6px !important;
        padding: 8px 18px !important;
        font-weight: 900 !important;
        color: #000000 !important;
        text-transform: uppercase !important;
        font-size: 13px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #FFE600 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        transform: translate(-1px, -1px) !important;
    }

    /* Radio Buttons & Labels High Contrast */
    [data-testid="stRadio"] label, [data-testid="stRadio"] div, [data-testid="stRadio"] span, [data-testid="stRadio"] p {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 14px !important;
    }

    /* Captions & General Text High Contrast */
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 13px !important;
    }

    /* Selectbox Labels & Text */
    [data-testid="stSelectbox"] label, [data-testid="stSelectbox"] div, [data-testid="stSelectbox"] span, [data-testid="stSelectbox"] p {
        color: #000000 !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3D Neo-Brutalist Scene Visualizer
def render_3d_neural_orb(loading_text="SYNTHESIZING NEURAL PROMPT..."):
    three_html = f"""
    <div id="three-container" style="width: 100%; height: 230px; display: flex; justify-content: center; align-items: center; position: relative; background: #FFE600; border: 3.5px solid #000000; box-shadow: 6px 6px 0px #000000; border-radius: 10px;">
      <canvas id="three-canvas" style="width: 100%; height: 100%; outline: none;"></canvas>
      <div style="position: absolute; bottom: 12px; font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 900; background: #000000; color: #FFE600; padding: 4px 14px; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; letter-spacing: 0.08em; text-transform: uppercase; border-radius: 4px;">
        {loading_text}
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
      const canvas = document.getElementById("three-canvas");
      const renderer = new THREE.WebGLRenderer({{ canvas: canvas, alpha: true, antialias: true }});
      renderer.setSize(canvas.clientWidth, canvas.clientHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
      camera.position.z = 4.2;

      // Outer Neo-Brutalist Black Wireframe
      const geometry = new THREE.IcosahedronGeometry(1.35, 1);
      const wireframeMat = new THREE.MeshBasicMaterial({{
        color: 0x000000,
        wireframe: true,
      }});
      const mesh = new THREE.Mesh(geometry, wireframeMat);
      scene.add(mesh);

      // Inner Solid Core
      const coreGeo = new THREE.SphereGeometry(0.72, 32, 32);
      const coreMat = new THREE.MeshStandardMaterial({{
        color: 0xffffff,
        roughness: 0.1,
        metalness: 0.1,
      }});
      const core = new THREE.Mesh(coreGeo, coreMat);
      scene.add(core);

      // Orbiting particles
      const particlesGeo = new THREE.BufferGeometry();
      const count = 120;
      const positions = new Float32Array(count * 3);
      for(let i = 0; i < count * 3; i += 3) {{
        const angle = (i / 3) * (Math.PI * 2 / count);
        const radius = 2.1 + (Math.random() - 0.5) * 0.4;
        positions[i] = Math.cos(angle) * radius;
        positions[i+1] = (Math.random() - 0.5) * 0.7;
        positions[i+2] = Math.sin(angle) * radius;
      }}
      particlesGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const particleMat = new THREE.PointsMaterial({{
        color: 0x000000,
        size: 0.08,
      }});
      const particleRing = new THREE.Points(particlesGeo, particleMat);
      scene.add(particleRing);

      const light1 = new THREE.PointLight(0xffffff, 2.0, 50);
      light1.position.set(5, 5, 5);
      scene.add(light1);
      const light2 = new THREE.AmbientLight(0xffffff, 1.2);
      scene.add(light2);

      function animate() {{
        requestAnimationFrame(animate);
        mesh.rotation.x += 0.012;
        mesh.rotation.y += 0.018;
        core.rotation.y -= 0.015;
        particleRing.rotation.y += 0.010;
        particleRing.rotation.x = Math.sin(Date.now() * 0.002) * 0.3;
        renderer.render(scene, camera);
      }}
      animate();
    </script>
    """
    components.html(three_html, height=240)

# Neo-Brutalist Hero Header
st.markdown("""
<div class='neo-hero-container'>
  <div class='neo-hero-badge'>⚡ GROQ MULTIMODAL VISION AI</div>
  <div class='neo-hero-title'>IMAGE TO <span>PROMPT</span></div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown("<span class='neo-section-label'>01 // Source Image</span>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "jfif"],
        label_visibility="collapsed"
    )

    if uploaded_image is not None:
        # Detect new image upload and reset analysis state
        if st.session_state.uploaded_file_name != uploaded_image.name:
            st.session_state.uploaded_file_name = uploaded_image.name
            st.session_state.image_rotation = 0
            st.session_state.analysis_result = None
            st.session_state.last_error = None

        try:
            pil_image = ImageOps.exif_transpose(Image.open(uploaded_image)).convert("RGB")
            if st.session_state.image_rotation != 0:
                pil_image = pil_image.rotate(-st.session_state.image_rotation, expand=True)
            st.image(pil_image, caption="", use_container_width=True)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            pil_image = None

        if pil_image is not None:
            # Generation Mode Selector
            st.markdown("<span class='neo-section-label'>02 // Generation Mode</span>", unsafe_allow_html=True)
            gen_mode = st.radio(
                "Prompt Mode",
                ["Mode 1: Exact Image Recreation", "Mode 2: Creative Prompt"],
                index=0,
                label_visibility="collapsed"
            )

            creative_controls = {}
            if gen_mode == "Mode 2: Creative Prompt":
                with st.expander("🎨 Creative Overrides & Controls", expanded=True):
                    c_style = st.selectbox("Aesthetic Style", ["Auto / Original", "Photorealistic", "Cyberpunk / Synthwave", "Studio Commercial", "Cinematic Film", "Anime / Manga", "3D Pixar / Unreal 5", "Oil Painting", "Flat Vector / SVG"], index=0)
                    c_light = st.selectbox("Lighting Setup", ["Auto / Original", "Golden Hour Sunlight", "Studio Softbox Lighting", "Cinematic Volumetric Beams", "Neon Glow / Cyberpunk", "Dramatic Chiaroscuro", "Flat Diffuse Light"], index=0)
                    c_cam = st.selectbox("Camera Perspective", ["Auto / Original", "Eye-Level 50mm Natural", "Wide-Angle 24mm Dramatic", "Macro Close-Up Extreme", "Drone Aerial Bird's-Eye", "Low-Angle Hero Shot"], index=0)
                    c_mood = st.selectbox("Atmospheric Mood", ["Auto / Original", "Vibrant & Energetic", "Serene & Peaceful", "Moody & Mysterious", "High-Tech & Futuristic", "Whimsical & Playful"], index=0)
                    creative_controls = {"style": c_style, "lighting": c_light, "camera": c_cam, "mood": c_mood}

            btn_c1, btn_c2, btn_c3 = st.columns([1.1, 0.9, 0.7])
            with btn_c1:
                analyze_clicked = st.button("⚡ GENERATE", type="primary", use_container_width=True)
            with btn_c2:
                rotate_clicked = st.button("↺ ROTATE", use_container_width=True)
            with btn_c3:
                clear_clicked = st.button("🗑️ CLEAR", use_container_width=True)

            if rotate_clicked:
                st.session_state.image_rotation = (st.session_state.image_rotation + 90) % 360
                st.session_state.analysis_result = None
                st.session_state.last_error = None
                st.rerun()

            if clear_clicked:
                st.session_state.analysis_result = None
                st.session_state.uploaded_file_name = None
                st.session_state.image_rotation = 0
                st.session_state.last_error = None
                st.rerun()

            st.caption(f"⚡ Active Engine: **Groq Vision ({st.session_state.groq_vision_model})**")

            if st.session_state.last_error:
                st.error(st.session_state.last_error)

            if analyze_clicked:
                groq_key = st.session_state.get("input_groq_key", "").strip() or st.session_state.groq_api_key.strip()

                if not groq_key or len(groq_key) < 10:
                    st.session_state.last_error = "❌ Groq API Key is required! Please enter your Groq API Key in Settings (⚙️ bottom left)."
                    st.session_state.analysis_result = None
                    st.rerun()

                with st.container():
                    render_3d_neural_orb("SENDING TO GROQ VISION API & ANALYZING PIXELS...")
                    time.sleep(0.3)

                    st.session_state.last_error = None
                    vision_result, err_detail = GroqService.analyze_with_vision(
                        pil_image,
                        groq_key,
                        model=st.session_state.groq_vision_model
                    )

                    if vision_result:
                        mode_name = "Exact Image Recreation" if gen_mode.startswith("Mode 1") else "Creative Prompt"
                        synthesized = PromptIntelligenceEngine.synthesize_from_vision_json(
                            vision_result,
                            mode=mode_name,
                            creative_controls=creative_controls
                        )
                        synthesized["engine_used"] = f"Groq Cloud AI ({st.session_state.groq_vision_model})"
                        st.session_state.analysis_result = synthesized
                        st.session_state.last_error = None
                    else:
                        st.session_state.last_error = f"❌ Groq Vision Error: {err_detail or 'Unable to analyze image with Groq Vision API. Please verify your Groq API key.'}"
                        st.session_state.analysis_result = None

                st.rerun()

    # Left-Down Settings Popover Modal (Zero homescreen clutter)
    with st.popover("⚙️ SETTINGS & GROQ API KEY", use_container_width=True):
        st.markdown("### 🛠️ GROQ VISION SETTINGS")
        st.caption("Configure Groq Cloud ultra-fast inference & multimodal vision model.")

        groq_input = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            key="input_groq_key",
            help="Get your free API key from console.groq.com/keys"
        )
        if groq_input.strip() != st.session_state.groq_api_key:
            st.session_state.groq_api_key = groq_input.strip()
            ConfigService.save_config({"groq_api_key": groq_input.strip()})

        selected_model = st.selectbox(
            "Groq Vision Model",
            [
                "qwen/qwen3.8-27b"
            ],
            index=0,
            key="select_groq_vision_model"
        )
        if selected_model != st.session_state.groq_vision_model:
            st.session_state.groq_vision_model = selected_model
            ConfigService.save_config({"groq_vision_model": selected_model})

        if st.button("🧪 TEST GROQ API KEY", use_container_width=True, key="btn_test_groq"):
            with st.spinner("Pinging Groq Cloud API..."):
                ok, msg = GroqService.test_connection(st.session_state.groq_api_key)
                st.session_state.groq_test_status = (ok, msg)
                if ok:
                    ConfigService.save_config({"groq_api_key": st.session_state.groq_api_key})

        if st.session_state.groq_test_status:
            ok, msg = st.session_state.groq_test_status
            if ok:
                st.success(msg)
            else:
                st.error(msg)

with col_right:
    st.markdown("<span class='neo-section-label'>03 // Visual Intelligence Output</span>", unsafe_allow_html=True)

    if uploaded_image is None:
        st.info("👈 Upload an image on the left to extract visual attributes and generate prompts.")
    elif st.session_state.analysis_result is None:
        st.info("👈 Image loaded! Click **⚡ GENERATE** on the left to analyze the scene and synthesize prompts.")
    elif st.session_state.analysis_result is not None:
        res = st.session_state.analysis_result

        # Domain & Engine Badges
        img_type = res.get('image_type', 'Photograph / natural scene')
        eng_used = res.get('engine_used', 'Multimodal Vision API')
        conf_val = res.get('confidence', 0.95)
        conf_str = f"{int(conf_val * 100)}%" if isinstance(conf_val, (int, float)) else str(conf_val)
        mode_used = res.get('mode_used', 'Exact Recreation')

        st.markdown(f"<span class='neo-pill'>🏷️ {img_type}</span> <span class='neo-pill-white'>⚡ {eng_used}</span> <span class='neo-pill'>🎯 {conf_str} CONFIDENCE</span> <span class='neo-pill-white'>🛠️ {mode_used}</span>", unsafe_allow_html=True)

        # Grounded Forensic Visual Description
        st.markdown("<span class='neo-section-label'>Grounded Visual Description</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='neo-detail-text'>{res['detailed_description']}</div>", unsafe_allow_html=True)

        # Detected Objects
        det_objs = res.get('detected_objects', [])
        if det_objs:
            st.markdown("<span class='neo-section-label'>Detected Objects & Elements</span>", unsafe_allow_html=True)
            obj_pills = "".join([f"<span class='neo-pill-white'>{obj}</span>" for obj in det_objs])
            st.markdown(f"<div style='margin-bottom: 1rem;'>{obj_pills}</div>", unsafe_allow_html=True)

        # Detected Text / OCR
        det_text = res.get('detected_text', [])
        if det_text:
            st.markdown("<span class='neo-section-label'>OCR & Visible Text Content</span>", unsafe_allow_html=True)
            text_pills = "".join([f"<span class='neo-pill'>{t}</span>" for t in det_text])
            st.markdown(f"<div style='margin-bottom: 1rem;'>{text_pills}</div>", unsafe_allow_html=True)

        # Generative Prompts Tabs
        st.markdown("<span class='neo-section-label'>Generative AI Prompts</span>", unsafe_allow_html=True)
        tab_master, tab_mj, tab_sd, tab_flux = st.tabs([
            "🌟 Master Reconstruction Prompt",
            "⛵ Midjourney v6",
            "🎨 Stable Diffusion SDXL",
            "🤖 FLUX / DALL-E 3"
        ])

        with tab_master:
            st.text_area("High-Fidelity Reconstruction Prompt", value=res["master_prompt"], height=140, key="ta_master", label_visibility="collapsed")
            st.download_button(
                label="📥 DOWNLOAD PROMPT (.TXT)",
                data=res["master_prompt"],
                file_name="reconstruction_prompt.txt",
                mime="text/plain",
                use_container_width=True
            )

        with tab_mj:
            st.code(res["midjourney_prompt"], language="text")
            st.caption("Paste directly into Discord `/imagine prompt:`.")

        with tab_sd:
            st.text_area("Positive Prompt", value=res["stable_diffusion_prompt"], height=100, key="ta_sd_pos")
            st.text_area("Negative Prompt", value=res["negative_prompt"], height=65, key="ta_sd_neg")

        with tab_flux:
            st.text_area("FLUX / DALL-E Prompt", value=res["master_prompt"], height=130, key="ta_flux", label_visibility="collapsed")

        # Structured Visual Breakdown
        with st.expander("🧬 STRUCTURED VISUAL ATTRIBUTES", expanded=True):
            attrs = res.get("attributes", {})
            for k, v in attrs.items():
                st.markdown(f"<span class='neo-pill'>{k}</span> {v}", unsafe_allow_html=True)

        # Raw Vision API JSON viewer
        raw_json = res.get("raw_vision_json")
        if raw_json:
            with st.expander("🔍 RAW VISION API STRUCTURED JSON", expanded=False):
                st.json(raw_json)

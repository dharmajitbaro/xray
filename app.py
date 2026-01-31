import streamlit as st
import os
import subprocess
import sys

# --- STEP 1: AUTO-INSTALL DETECTRON2 ---
# This block runs before imports to ensure the environment is ready
@st.cache_resource
def install_dependencies():
    try:
        import detectron2
    except ImportError:
        st.info("Setting up Detectron2 environment... This takes 1-2 minutes.")
        # Install the pre-built CPU wheel for Linux
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "detectron2", "-f", 
            "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html"
        ])
        st.success("Setup complete! Rerunning app...")
        st.rerun()

install_dependencies()

# --- STEP 2: IMPORTS ---
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# --- STEP 3: CONFIGURATION ---
st.set_page_config(page_title="Fracture Detection AI", layout="wide")
CLASS_NAMES = ["Fracture"] 
MODEL_PATH = "output_xray/model_final.pth"

@st.cache_resource
def load_fracture_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Critical Error: Model file not found at {MODEL_PATH}")
        return None, None
        
    cfg = get_cfg()
    # Ensure this matches your training architecture (e.g., Faster R-CNN)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cpu"  # Required for Streamlit Cloud
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg), cfg

# --- STEP 4: UI AND INFERENCE ---
st.title("🦴 X-Ray Fracture Detection System")
st.write("M.Tech AI Project Deployment")

predictor, cfg = load_fracture_model()
metadata = MetadataCatalog.get("xray_data").set(thing_classes=CLASS_NAMES)

uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file and predictor:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Resize image to prevent Out-of-Memory (OOM) errors on Streamlit Cloud
    h, w = image.shape[:2]
    if w > 800:
        image = cv2.resize(image, (800, int(h * 800 / w)))

    with st.spinner("Analyzing X-ray..."):
        outputs = predictor(image)
        
        # Visualize detections
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Display
        st.image(out.get_image()[:, :, ::-1], caption="Detection Results", use_column_width=True)
        
        # Results Table
        num_fractures = len(outputs["instances"])
        if num_fractures > 0:
            st.warning(f"Detected {num_fractures} potential fracture(s).")
        else:
            st.success("No fractures detected with current confidence settings.")

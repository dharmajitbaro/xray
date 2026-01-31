import streamlit as st
import os
import subprocess
import sys

# --- STEP 1: FAIL-SAFE INSTALL ---
@st.cache_resource
def install_detectron2():
    try:
        import detectron2
    except ImportError:
        st.info("Installing Detectron2 for Python 3.10... Please wait 2-3 minutes.")
        # We use a direct link to the wheel to avoid "Forbidden" or "Not Found" errors
        wheel_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl"
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
        st.success("Detectron2 installed successfully!")
        st.rerun()

install_detectron2()

# --- STEP 2: IMPORTS ---
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# --- STEP 3: MODEL LOADING ---
st.title("🦴 Fracture Detection System")
MODEL_PATH = "output_xray/model_final.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Check your folder structure!")
        return None
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Change if you have more than one class
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg)

predictor = load_model()

# --- STEP 4: INFERENCE ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and predictor:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Resize to prevent RAM crashes
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
    
    with st.spinner('Detecting...'):
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("xray_data"), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], caption='Processed Image', use_column_width=True)

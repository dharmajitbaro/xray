import streamlit as st
import cv2
import numpy as np
import torch
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# --- APP CONFIG ---
st.set_page_config(page_title="X-Ray Fracture Detection", layout="centered")
CLASS_NAMES = ["Fracture"] 
MODEL_PATH = "output_xray/model_final.pth"

@st.cache_resource
def load_model():
    cfg = get_cfg()
    # Ensure this matches your training architecture (e.g., Faster R-CNN)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cpu"  # Mandatory for Streamlit Cloud
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg), cfg

# --- UI ---
st.title("🦴 X-Ray Fracture Detection System")
st.info("M.Tech AI Project: Detecting bone fractures using Detectron2")

if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Could not find {MODEL_PATH}. Please ensure it is in your GitHub repo.")
else:
    predictor, cfg = load_model()
    metadata = MetadataCatalog.get("xray_val").set(thing_classes=CLASS_NAMES)

    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Resize for stability (prevents RAM crashes)
        h, w = img.shape[:2]
        if w > 1000:
            img = cv2.resize(img, (1000, int(h * 1000 / w)))

        with st.spinner("Analyzing Image..."):
            outputs = predictor(img)
            
            # Draw results
            v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
            # Layout results
            st.image(out.get_image()[:, :, ::-1], caption="Detection Results", use_column_width=True)
            
            num_instances = len(outputs["instances"])
            if num_instances > 0:
                st.warning(f"Found {num_instances} potential fracture areas.")
            else:
                st.success("No fractures detected at the current threshold.")

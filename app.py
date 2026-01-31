import streamlit as st
import os
import subprocess
import sys

# --- STEP 1: INSTALL ---
@st.cache_resource
def install_detectron2():
    try:
        import detectron2
    except ImportError:
        st.info("Installing Detectron2... Please wait.")
        # This wheel matches Python 3.10 perfectly
        wheel = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl"
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel])
        st.rerun()

install_detectron2()

# --- STEP 2: APP ---
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

st.title("🦴 Fracture Detection")

@st.cache_resource
def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.MODEL.WEIGHTS = "output_xray/model_final.pth"
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

if os.path.exists("output_xray/model_final.pth"):
    predictor = get_predictor()
    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Inference
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("xray"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1])
else:
    st.error("Model file not found in output_xray/model_final.pth")

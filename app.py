import streamlit as st
import cv2
import numpy as np
import torch
import os

# --- DETECTRON2 IMPORTS ---
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
except ImportError:
    st.error("Detectron2 is not installed. Please check your requirements.txt.")

# --- CONFIGURATION ---
CLASS_NAMES = ["Fracture"]  # Adjust if you have more classes
MODEL_WEIGHTS_PATH = "output_xray/model_final.pth"

# Page Layout
st.set_page_config(page_title="X-Ray Fracture Detection", layout="wide")

@st.cache_resource
def load_predictor():
    """Loads the Detectron2 model and caches it to memory."""
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        st.error(f"Model file not found at {MODEL_WEIGHTS_PATH}. Please check your GitHub folder structure.")
        return None, None

    cfg = get_cfg()
    
    # Use the same base config you used during training
    # For Faster R-CNN R50 FPN:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
    
    # CRITICAL: Streamlit Cloud uses CPU
    cfg.MODEL.DEVICE = "cpu" 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold
    
    return DefaultPredictor(cfg), cfg

# --- UI INTERFACE ---
st.title("🦴 AI-Powered X-Ray Fracture Detection")
st.markdown("""
This application uses a **Detectron2** model trained to identify fractures in X-ray images. 
Upload an image below to see the predictions.
""")

# Sidebar settings
st.sidebar.header("Settings")
thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Load model
predictor, cfg = load_predictor()

# Register Metadata
metadata = MetadataCatalog.get("fracture_data").set(thing_classes=CLASS_NAMES)

uploaded_file = st.file_uploader("Upload an X-ray image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, channels="BGR", use_column_width=True)
        
    with col2:
        st.subheader("Model Prediction")
        if predictor:
            with st.spinner("Analyzing..."):
                # Update threshold based on slider
                predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
                
                # Run Inference
                outputs = predictor(image)
                
                # Visualize
                v = Visualizer(image[:, :, ::-1], 
                               metadata=metadata, 
                               scale=1.0, 
                               instance_mode=ColorMode.SEGMENTATION)
                
                instances = outputs["instances"].to("cpu")
                out = v.draw_instance_predictions(instances)
                
                # Display processed image
                st.image(out.get_image()[:, :, ::-1], use_column_width=True)
                
                # Summary results
                num_fractures = len(instances)
                if num_fractures > 0:
                    st.warning(f"Detected {num_fractures} potential fracture(s).")
                else:
                    st.success("No fractures detected.")
        else:
            st.error("Model failed to load.")

st.divider()
st.info("Note: This tool is for educational purposes as part of an M.Tech project and should not be used for clinical diagnosis.")

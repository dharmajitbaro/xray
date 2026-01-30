import streamlit as st
import cv2
import numpy as np
from predictor import load_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

st.set_page_config(page_title="X-ray Fracture Detection", layout="centered")

st.title("X-ray Fracture Detection")
st.write("Upload an X-ray image to detect possible fractures.")

@st.cache_resource
def load_predictor():
    return load_model()

predictor = load_predictor()
metadata = MetadataCatalog.get("xray_train")

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded X-ray", channels="BGR")

    with st.spinner("Running fracture detection..."):
        outputs = predictor(image)

    instances = outputs["instances"].to("cpu")
    num_detections = len(instances)

    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=0.8
    )
    out = v.draw_instance_predictions(instances)

    st.image(
        out.get_image()[:, :, ::-1],
        caption=f"Detected fractures: {num_detections}",
        use_column_width=True
    )

    st.write("### Detection Details")
    st.write(instances)
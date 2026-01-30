import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = "output_xray/model_final.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    return predictor
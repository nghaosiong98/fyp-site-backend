from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import torch
import numpy as np
import cv2
import os


class AlgaeModel:
    def __init__(self):
        self.model = load_model(os.path.join('model', 'algae_model'))

    def preprocess_image(self, image):
        preprocessed_image = cv2.resize(image, (224, 224))
        preprocessed_image = preprocess_input(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        return preprocessed_image

    def predict(self, image):
        preprocessed_image = self.preprocess_image(image)
        (o1, o2) = self.model.predict(preprocessed_image)[0]
        label = 'Eutrophic' if o1 > o2 else 'Not Eutrophic'
        return o1, o2, label


class WaterModel:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        model = os.path.join('model', 'water_model', 'model_final.pth')

        if os.path.isfile(model):
            print('[INFO] Using trained model {}'.format(model), flush=True)
        else:
            print('[WARNING] No model found at {}'.format(model, flush=True))
            model = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.WEIGHTS = model
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

        if torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cuda"
        else:
            self.cfg.MODEL.DEVICE = "cpu"

    def inference(self, image):
        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(image)
        box = np.asarray(outputs['instances'].pred_boxes.tensor.cpu().numpy()[0], dtype=int)
        (h, w) = image.shape[:2]
        print(box)
        (start_x, start_y, end_x, end_y) = box
        (start_x, start_y) = (max(0, start_x), max(0, start_y))
        (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))
        cropped = image[start_x:end_x, start_y:end_y]
        return cropped

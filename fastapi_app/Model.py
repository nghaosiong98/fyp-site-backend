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

    def predict(self, image_string):
        nparr = np.fromstring(image_string, np.uint8)
        raw_image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        preprocessed_image = self.preprocess_image(raw_image)
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

    def inference(self, img_str):
        predictor = DefaultPredictor(self.cfg)
        nparr = np.frombuffer(img_str, 'u1')
        im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        outputs = predictor(im)
        mask = np.asarray(outputs['instances'].pred_masks.cpu().numpy()[0], dtype=np.uint8)
        cropped = cv2.bitwise_and(im, im, mask=mask)
        return cropped

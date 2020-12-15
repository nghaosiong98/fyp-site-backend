from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2


class AlgaeModel:
    def __init__(self):
        self.model = load_model('model')

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
        return o1, o2

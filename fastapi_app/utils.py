import numpy as np
import cv2


def byte_to_img(image_string):
    nparr = np.fromstring(image_string, np.uint8)
    raw_image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    return raw_image

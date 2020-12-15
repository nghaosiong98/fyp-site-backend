from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import werkzeug
import numpy as np
import cv2


class Predict(Resource):
    def __init__(self):
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument("images", type=werkzeug.datastructures.FileStorage, location='files', action='append',
                            required=True, help='please select at least 1 images')
        parser.add_argument("lat", required=True, help="lat is missing")
        parser.add_argument("lng", required=True, help="lng is missing")
        self.req_parser = parser

        self.model = load_model('model')

    def classify_image(self, image):
        preprocessed_image = cv2.resize(image, (224, 224))
        preprocessed_image = preprocess_input(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        (o1, o2) = self.model.predict(preprocessed_image)[0]
        return o1, o2

    def post(self):
        lat = self.req_parser.parse_args(strict=True).get("lat", None)
        lng = self.req_parser.parse_args(strict=True).get("lng", None)
        images = self.req_parser.parse_args(strict=True).get("images", None)
        server.logger.info("Number of imager")
        responses = []
        for image in images:
            filename = image.filename
            image_string = image.read()
            nparr = np.fromstring(image_string, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
            o1, o2 = self.classify_image(img)
            result = {
                'filename': filename,
                'result': {
                    0: float(o1),
                    1: float(o2),
                },
                'location': {
                    'lat': float(lat),
                    'lng': float(lng),
                },
            }
            responses.append(result)

        return jsonify(responses)

    def get(self):
        return "API is online."


server = Flask(__name__)
CORS(server)
api = Api(server)

api.add_resource(Predict, '/predict')


@server.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

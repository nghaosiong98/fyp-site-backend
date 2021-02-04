from typing import List
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from Model import AlgaeModel, WaterModel
from utils import byte_to_img, upload_blob_with_metadata

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*', 'https://fyp.haosiongng.com', 'http://localhost:3000'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
]

app = FastAPI(middleware=middleware)
model = AlgaeModel()
water_model = WaterModel()


@app.get("/")
def test_root():
    return {"msg": "OK"}


@app.post("/")
def test_root_post():
    headers = {
        "Access-Control-Allow-Origin": "*",
    }
    return JSONResponse(content={"msg": "OK"}, headers=headers)


@app.get("/predict")
def test_api():
    return {"msg": "OK"}


@app.post("/predict")
def predict(images: List[UploadFile] = File(...), lat: float = Body(...), lng: float = Body(...)):
    response = {
        'results': [],
        'lat': float(lat),
        'lng': float(lng),
    }
    for image in images:
        byte_string = image.file.read()
        im = byte_to_img(byte_string)
        cropped_im = water_model.inference(im)
        if cropped_im is None:
            response['results'].append({
                'filename': image.filename,
                'prediction': {
                    0: None,
                    1: None,
                },
                'label': -1
            })
            continue
        o1, o2, label = model.predict(cropped_im)
        metadata = {
            'predict_0': float(o1),
            'predict_1': float(o2),
            'label': int(label),
            'lat': float(lat),
            'lng': float(lng),
        }
        upload_blob_with_metadata("fyp-backend-image", byte_string, image.filename, metadata)
        response['results'].append({
            'filename': image.filename,
            'prediction': {
                0: float(o1),
                1: float(o2),
            },
            'label': int(label),
        })
    headers = {
        "Access-Control-Allow-Origin": "*",
    }
    return JSONResponse(content=response, headers=headers)

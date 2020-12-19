from typing import List
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from Model import AlgaeModel

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

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*', 'https://fyp.haosiongng.com', 'http://localhost:3000'],
#     allow_credentials=True,
#     allow_methods=['*'],
#     allow_headers=['*'],
# )


@app.get("/predict")
def test_api():
    return {"status": "OK"}


@app.post("/predict")
def predict(images: List[UploadFile] = File(...), lat: float = Body(...), lng: float = Body(...)):
    response = {
        'results': [],
        'lat': float(lat),
        'lng': float(lng),
    }
    for image in images:
        o1, o2, label = model.predict(image.file.read())
        response['results'].append({
            'filename': image.filename,
            'prediction': {
                0: float(o1),
                1: float(o2),
            },
            'label': str(label),
        })
    headers = {
        "Access-Control-Allow-Origin": "*",
    }
    return JSONResponse(content=response, headers=headers)

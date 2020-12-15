from typing import List
from fastapi import FastAPI, File, UploadFile, Body
from Model import AlgaeModel

app = FastAPI()
model = AlgaeModel()


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
        o1, o2 = model.predict(image.file.read())
        response['results'].append({
            'filename': image.filename,
            'prediction': {
                0: float(o1),
                1: float(o2),
            }
        })
    return response

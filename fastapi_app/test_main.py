from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_main():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'msg': 'OK'}


def test_prediction():
    response = client.post('/predict',
                           files={'images': open('test_images/room.jpg', 'rb'),
                                  'images': open('test_images/lake.jpg', 'rb')},
                           data={'lat': 0.01, 'lng': 0.01})
    assert response.status_code == 200

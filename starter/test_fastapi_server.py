from fastapi.testclient import TestClient
from fastapi_server import app

client = TestClient(app)

def test_home():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the FastAPI model"}

def test_predict_1():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "educationnum": 13,
        "maritalstatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hoursperweek": 40,
        "nativecountry": "United-States"
    }
    r = client.post('/inference', json=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": '<=50K'}

def test_predict_2():
    data = {
        "age": 40,
        "workclass": "Private",
        "education": "Doctorate",
        "educationnum": 16,
        "maritalstatus": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursperweek": 60,
        "nativecountry": "United-States"
    }
    r = client.post('/inference', json=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": '>50K'}
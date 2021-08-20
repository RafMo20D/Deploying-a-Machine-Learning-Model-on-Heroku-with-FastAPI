import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import starter.helper_function
from starter.helper_function import inference
import pandas as pd
import numpy as np

class Data(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    educationnum: int
    maritalstatus: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    hoursperweek: int
    nativecountry: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -q") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")

app = FastAPI()

@app.get("/")
def home():
    return {"greeting": "Welcome to the FastAPI model"}


@app.post('/inference')
async def predict_income(inputdata: Data):
    model = joblib.load("model/trainedmodel.joblib")
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/lb.joblib")

    input_data = np.array([[inputdata.age,
                            inputdata.workclass,
                            inputdata.education,
                            inputdata.educationnum,
                            inputdata.maritalstatus,
                            inputdata.occupation,
                            inputdata.relationship,
                            inputdata.race,
                            inputdata.sex,
                            inputdata.hoursperweek,
                            inputdata.nativecountry
                            ]])    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    df = pd.DataFrame(input_data, columns=[
        "age",
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = starter.helper_function.process_data(
                df,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    prediction = inference(model, X)
    y = lb.inverse_transform(prediction)[0]
    return {"prediction": y}
    
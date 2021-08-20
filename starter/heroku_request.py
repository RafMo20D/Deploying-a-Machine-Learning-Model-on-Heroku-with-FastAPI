import requests

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

response = requests.post(
    url='https://vin-project3-app.herokuapp.com/inference',
    json=data
)

print(response.status_code)
print(response.json())
import requests

# GET
r = requests.get("http://127.0.0.1:8000/")
print("Status Code:", r.status_code)
print("Result:", r.json())

# POST
payload = {
    "age": 39,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
r = requests.post("http://127.0.0.1:8000/predict/", json=payload)
print("Status Code:", r.status_code)
print("Result:", r.json())


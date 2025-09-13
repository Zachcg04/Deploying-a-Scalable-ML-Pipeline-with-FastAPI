import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# ---------------------------
# Input data schema (unchanged)
# ---------------------------
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# ---------------------------
# Load trained artifacts
# ---------------------------
encoder_path = os.path.join("model", "encoder.pkl")
model_path = os.path.join("model", "model.pkl")

encoder = load_model(encoder_path)
model = load_model(model_path)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

# Root GET
@app.get("/")
async def get_root():
    return {"message": "Welcome to the Census ML API!"}

# Inference POST
@app.post("/predict/")
async def post_inference(data: Data):
    # Turn request into DataFrame
    data_dict = data.dict()
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(data)

    # Categorical features
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

    # Process input row
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,
    )

    # Run inference
    preds = inference(model, X)

    return {"prediction": apply_label(preds)}


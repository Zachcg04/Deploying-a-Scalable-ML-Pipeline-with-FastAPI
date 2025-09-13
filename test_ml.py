import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference

def test_train_model():
    data = pd.read_csv("data/census.csv")
    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary", training=True
    )
    model = train_model(X_train, y_train)
    assert model is not None

def test_inference_shape():
    data = pd.read_csv("data/census.csv")
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary", training=True
    )
    model = train_model(X_train, y_train)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary", training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    assert len(preds) == len(y_test)

def test_labels_binary():
    data = pd.read_csv("data/census.csv")
    assert set(data["salary"].unique()) <= {">50K", "<=50K"}


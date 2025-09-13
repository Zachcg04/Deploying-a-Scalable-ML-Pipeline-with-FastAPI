import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data
project_path = os.getcwd()  # assumes you run from project root
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Split data into train/test
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Define categorical features
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

# Process train/test sets
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save the model and encoder
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)

# Load model back (just to verify)
model = load_model(model_path)

# Run inference on test data
preds = inference(model, X_test)

# Calculate overall metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Overall -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute slice performance and save to file
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test,
                column_name=col,
                slice_value=slice_value,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
            )
            f.write(f"{col}: {slice_value}, Count: {count:,}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n\n")


import json
import os

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def load_data():
    """Load training and validation data using default paths."""
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]
    train_data = pd.read_csv("./temp/train.csv", names=column_names, header=None)
    validation_data = pd.read_csv("./temp/test.csv", names=column_names, header=None)
    return train_data, validation_data


def preprocess_data(train_data, validation_data):
    """Preprocess data by encoding labels and creating DMatrix objects."""
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["target"])
    y_validation = label_encoder.transform(validation_data["target"])

    X_train = train_data.drop("target", axis=1)
    X_validation = validation_data.drop("target", axis=1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    return dtrain, dvalidation


def train_model(dtrain, dvalidation):
    """Train the XGBoost model using default hyperparameters."""
    hyperparameters = {
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weight": 6,
        "subsample": 0.8,
        "verbosity": 1,
        "objective": "multi:softmax",
        "tree_method": "gpu_hist",
        "predictor": "auto",
        "num_class": 3,
    }
    watchlist = [(dtrain, "train"), (dvalidation, "validation")]
    model = xgb.train(
        hyperparameters,
        dtrain,
        num_boost_round=100,
        evals=watchlist,
        early_stopping_rounds=10,
    )
    return model


def save_model(model, model_dir="model"):
    """Save the trained model and hyperparameters to the specified directory."""
    os.makedirs(model_dir, exist_ok=True)
    model_location = os.path.join(model_dir, "xgboost-model")
    model.save_model(model_location)

    hyperparameters_location = os.path.join(model_dir, "hyperparameters.json")
    with open(hyperparameters_location, "w") as f:
        json.dump(model.save_config(), f)

    print(f"Model saved to {model_location}")


def execute_training_pipeline():
    """Execute the training pipeline with default values."""
    # Load data
    train_data, validation_data = load_data()

    # Preprocess data
    dtrain, dvalidation = preprocess_data(train_data, validation_data)

    # Train the model
    model = train_model(dtrain, dvalidation)

    # Save the model
    save_model(model)


if __name__ == "__main__":
    execute_training_pipeline()

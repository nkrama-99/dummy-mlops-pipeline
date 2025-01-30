import json
import os

import boto3
import pandas as pd
import xgboost as xgb
from prefect import flow, task
from sklearn.preprocessing import LabelEncoder


@task(log_prints=True)
def load_data():
    """Load training and validation data from S3 bucket."""

    s3_client = boto3.client("s3")

    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    # Replace with your actual bucket name and file paths
    bucket_name = "dummy-data"
    train_file_key = "train.csv"
    validation_file_key = "test.csv"

    # Download the files from S3 to a temporary location
    s3_client.download_file(bucket_name, train_file_key, "train.csv")
    s3_client.download_file(bucket_name, validation_file_key, "test.csv")

    # Load the data into pandas DataFrame
    train_data = pd.read_csv("train.csv", names=column_names, header=None)
    validation_data = pd.read_csv("test.csv", names=column_names, header=None)

    return train_data, validation_data


@task(log_prints=True)
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


@task(log_prints=True)
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


@task(log_prints=True)
def save_model(model, model_dir="model"):
    """Save the trained model and hyperparameters to S3 bucket."""
    s3_client = boto3.client("s3")

    # Set your S3 bucket name and paths for saving the model
    bucket_name = "dummy-model"
    model_file_key = "xgboost-model"
    hyperparameters_file_key = "hyperparameters.json"

    # Save the model to a local file first
    model_location = os.path.join(model_dir, "xgboost-model")
    model.save_model(model_location)

    # Save the hyperparameters to a local file
    hyperparameters_location = os.path.join(model_dir, "hyperparameters.json")
    with open(hyperparameters_location, "w") as f:
        json.dump(model.save_config(), f)

    # Upload the model and hyperparameters to S3
    s3_client.upload_file(model_location, bucket_name, model_file_key)
    s3_client.upload_file(
        hyperparameters_location, bucket_name, hyperparameters_file_key
    )

    print(f"Model and hyperparameters saved to S3 bucket {bucket_name}")


@flow(log_prints=True, name="dummy-training-flow")
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
    flow.from_source(
        source="https://github.com/nkrama-99/dummy-mlops-pipeline.git",
        entrypoint="train.py:execute_training_pipeline",
    ).deploy(name="dummy-training-deployment", work_pool_name="dummy-worker-pool")

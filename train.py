import uuid

import pandas as pd
import xgboost as xgb
from prefect import flow, task
from prefect.blocks.system import Secret
from prefect.cache_policies import NONE
from prefect_aws import AwsCredentials, S3Bucket
from sklearn.preprocessing import LabelEncoder
from supabase import create_client


@task(log_prints=True, cache_policy=NONE)
def load_data(dataset_uuid):
    """Load training and validation data from S3 bucket."""
    print("Loading data from S3 bucket...")
    aws_credentials = AwsCredentials.load("aws-credentials")

    s3_bucket = S3Bucket(bucket_name="dummy-data-rama-432", credentials=aws_credentials)

    # Download the files from S3 to a temporary location
    print("Downloading train.csv and test.csv from S3...")
    s3_bucket.download_object_to_path("train.csv", "train.csv")
    s3_bucket.download_object_to_path("test.csv", "test.csv")

    # Load the data into pandas DataFrame
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    print("Loading training and validation data into pandas DataFrame...")
    train_data = pd.read_csv("train.csv", names=column_names, header=None)
    validation_data = pd.read_csv("test.csv", names=column_names, header=None)

    print(
        f"Data loaded: {len(train_data)} training samples and {len(validation_data)} validation samples."
    )

    s3_bucket.move_object("train.csv", f"{dataset_uuid}/train.csv")
    s3_bucket.move_object("test.csv", f"{dataset_uuid}/test.csv")

    return train_data, validation_data


@task(log_prints=True, cache_policy=NONE)
def preprocess_data(train_data, validation_data):
    """Preprocess data by encoding labels and creating DMatrix objects."""
    print("Preprocessing data: Encoding labels and creating DMatrix objects...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["target"])
    y_validation = label_encoder.transform(validation_data["target"])

    X_train = train_data.drop("target", axis=1)
    X_validation = validation_data.drop("target", axis=1)

    print("Data preprocessing completed. Creating DMatrix for XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)

    print(
        f"Preprocessing completed. Training data shape: {X_train.shape}, Validation data shape: {X_validation.shape}"
    )
    return dtrain, dvalidation


@task(log_prints=True, cache_policy=NONE)
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

    print("hyperparameters:", hyperparameters)
    print("Training XGBoost model...")
    watchlist = [(dtrain, "train"), (dvalidation, "validation")]
    model = xgb.train(
        hyperparameters,
        dtrain,
        num_boost_round=100,
        evals=watchlist,
        early_stopping_rounds=10,
    )

    print(f"Model training completed with {model.best_iteration} boosting rounds.")
    return model


@task(log_prints=True, cache_policy=NONE)
def save_model(model, model_uuid):
    """Save the trained model to S3 bucket."""
    print("Saving trained model...")
    # Save the model to a local file first
    model.save_model("xgboost-model")

    # Save the model to an S3 bucket
    aws_credentials = AwsCredentials.load("aws-credentials")

    s3_bucket = S3Bucket(
        bucket_name="dummy-model-rama-432", credentials=aws_credentials
    )

    print("Uploading model to S3...")
    s3_bucket.upload_from_path("xgboost-model", f"{model_uuid}-xgboost-model")

    print("Model saved to S3 successfully.")


@flow(log_prints=True, name="dummy-training-flow")
def generate_training_uuids():
    model_uuid = str(uuid.uuid4())
    dataset_uuid = str(uuid.uuid4())
    print(f"model_uuid: {model_uuid}")
    print(f"dataset_uuid: {dataset_uuid}")
    return model_uuid, dataset_uuid


@flow(log_prints=True, name="dummy-training-flow")
def add_model_to_registry(model_uuid, dataset_uuid):
    supabase = create_client(
        Secret.load("supabase-url").get(),
        Secret.load("supabase-key").get(),
    )

    supabase.table("model_registry").insert(
        {
            "dataset_uuid": dataset_uuid,
            "model_uuid": model_uuid,
            "is_tested": False,
        }
    ).execute()

    print(f"Model {model_uuid} added to registry")


@flow(log_prints=True, name="dummy-training-flow")
def execute_training_pipeline():
    """Execute the training pipeline with default values."""
    print("Starting the training pipeline...")

    model_uuid, dataset_uuid = generate_training_uuids()

    # Load data
    train_data, validation_data = load_data(dataset_uuid)

    # Preprocess data
    dtrain, dvalidation = preprocess_data(train_data, validation_data)

    # Train the model
    model = train_model(dtrain, dvalidation)

    # Save the model
    save_model(model, model_uuid)

    # Log model to registry
    add_model_to_registry(model_uuid, dataset_uuid)

    print("Training pipeline executed successfully.")


if __name__ == "__main__":
    print("Deploying training pipeline...")
    flow.from_source(
        source="https://github.com/nkrama-99/dummy-mlops-pipeline.git",
        entrypoint="train.py:execute_training_pipeline",
    ).deploy(name="dummy-training-deployment", work_pool_name="dummy-worker-pool")
    print("Training pipeline deployed successfully.")

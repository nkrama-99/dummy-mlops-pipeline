import uuid

import numpy as np
import xgboost as xgb
from prefect import flow, task
from prefect.blocks.system import Secret
from prefect.cache_policies import NONE
from prefect_aws import AwsCredentials, S3Bucket
from sklearn.metrics import accuracy_score
from supabase import create_client


@task(log_prints=True, cache_policy=NONE)
def load_model() -> xgb.Booster:
    """Load a saved XGBoost model from S3"""

    aws_credentials = AwsCredentials.load("aws-credentials")

    s3_bucket = S3Bucket(
        bucket_name="dummy-model-rama-432", credentials=aws_credentials
    )

    s3_bucket.download_object_to_path("xgboost-model", "xgboost-model")

    model = xgb.Booster()
    model.load_model("xgboost-model")

    return model


@task(log_prints=True, cache_policy=NONE)
def predict(model, samples) -> np.ndarray:
    """Make predictions using the loaded model.
    Args:
        model: Loaded XGBoost model.
        X: Features array/matrix in the same format used during training.
        feature_names: List of feature names corresponding to the columns in X.
    """

    # Define feature names (must match the training data)
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Convert samples to a NumPy array
    X = np.array(samples)

    # Convert input to DMatrix and include feature names
    dtest = xgb.DMatrix(X, feature_names=feature_names)

    # Get predictions
    predictions = model.predict(dtest)
    return predictions


@task(log_prints=True, cache_policy=NONE)
def validate_predictions(predictions, expected):
    """Validate predictions by comparing with expected values."""

    # Convert predictions to the closest class if necessary
    predicted_classes = np.round(predictions)  # Adjust this based on model output

    accuracy = accuracy_score(expected, predicted_classes)

    print(f"Validation Accuracy: {accuracy:.2%}")

    return accuracy


@task(log_prints=True, cache_policy=NONE)
def save_testing_details(accuracy):
    """Save accuracy details to Supabase."""
    supabase = create_client(
        Secret.load("supabase-url").get(),
        Secret.load("supabase-key").get(),
    )

    model_uuid = str(uuid.uuid4())

    supabase.table("model_accuracy").insert(
        {"model_uuid": model_uuid, "model_accuracy": accuracy}
    ).execute()

    print(f"New model saved: {model_uuid}")


@flow(log_prints=True, name="dummy-testing-flow")
def execute_testing_pipeline() -> None:
    samples = [
        [5.0, 3.4, 1.5, 0.2],
        [6.4, 3.2, 4.5, 1.5],
        [7.2, 3.6, 6.1, 2.5],
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 3.0, 5.8, 2.2],
        [7.6, 3.0, 6.6, 2.1],
    ]

    expected_labels = [0, 1, 2, 0, 0, 1, 1, 2, 2]

    model = load_model()

    predictions = predict(model, samples)

    for sample, prediction in zip(samples, predictions):
        print(f"Prediction for sample {sample}: {prediction}")

    accuracy = validate_predictions(predictions, expected_labels)


if __name__ == "__main__":
    print("Deploying testing pipeline...")
    flow.from_source(
        source="https://github.com/nkrama-99/dummy-mlops-pipeline.git",
        entrypoint="test.py:execute_testing_pipeline",
    ).deploy(name="dummy-testing-deployment", work_pool_name="dummy-worker-pool")
    print("Testing pipeline deployed successfully.")

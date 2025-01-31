from typing import Union

import numpy as np
import xgboost as xgb
from prefect import flow, task
from prefect.cache_policies import NONE
from prefect_aws import AwsCredentials, S3Bucket


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
def predict(model: xgb.Booster, X: Union[list[list[float]], np.ndarray]) -> np.ndarray:
    """Make predictions using the loaded model
    Args:
        model: Loaded XGBoost model
        X: Features array/matrix in the same format used during training
    """
    # Convert input to DMatrix (optional but recommended)
    dtest = xgb.DMatrix(np.array(X))
    # Get predictions
    predictions = model.predict(dtest)
    return predictions


@flow(log_prints=True, name="dummy-testing-flow")
def execute_testing_pipeline() -> None:
    samples = [[5.0, 3.4, 1.5, 0.2], [6.4, 3.2, 4.5, 1.5], [7.2, 3.6, 6.1, 2.5]]

    model = load_model()

    predictions = predict(model, samples)

    for sample, prediction in zip(samples, predictions):
        print(f"Prediction for sample {sample}: {prediction}")


if __name__ == "__main__":
    print("Deploying testing pipeline...")
    flow.from_source(
        source="https://github.com/nkrama-99/dummy-mlops-pipeline.git",
        entrypoint="test.py:execute_testing_pipeline",
    ).deploy(name="dummy-testing-deployment", work_pool_name="dummy-worker-pool")
    print("Testing pipeline deployed successfully.")

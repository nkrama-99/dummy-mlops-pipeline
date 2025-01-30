import argparse
import json
import os

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--min_child_weight", type=float, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--objective", type=str, default="multi:softmax")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--tree_method", type=str, default="gpu_hist")
    parser.add_argument("--predictor", type=str, default="auto")
    parser.add_argument("--num_class", type=int, default=3)

    # Data paths
    parser.add_argument("--train", type=str, default="./temp/train.csv")
    parser.add_argument("--validation", type=str, default="./temp/test.csv")
    parser.add_argument("--model_dir", type=str, default="model")

    args = parser.parse_args()

    # Load training and validation data
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]
    train_data = pd.read_csv(args.train, names=column_names, header=None)
    validation_data = pd.read_csv(args.validation, names=column_names, header=None)

    # Convert text labels to numeric values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["target"])
    y_validation = label_encoder.transform(validation_data["target"])

    # Get features
    X_train = train_data.drop("target", axis=1)
    X_validation = validation_data.drop("target", axis=1)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)

    hyperparameters = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        "objective": args.objective,
        "tree_method": args.tree_method,
        "predictor": args.predictor,
        "num_class": args.num_class,
    }

    # Train the model
    watchlist = [(dtrain, "train"), (dvalidation, "validation")]
    model = xgb.train(
        hyperparameters,
        dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        early_stopping_rounds=10,
    )

    # Save the model
    os.makedirs(args.model_dir, exist_ok=True)
    model_location = os.path.join(args.model_dir, "xgboost-model")
    model.save_model(model_location)

    # Save the model parameters
    hyperparameters_location = os.path.join(args.model_dir, "hyperparameters.json")
    with open(hyperparameters_location, "w") as f:
        json.dump(hyperparameters, f)

    print(f"Model saved to {model_location}")
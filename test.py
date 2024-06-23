import argparse
import pandas as pd
from data.pre_process_v2 import Dataset
from models import xgboost, logistic_regression, neural_network, random_forest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-M",
        "--model_type",
        type=str,
        required=True,
        choices=["LR", "XGB", "NN", "RF"],
    )
    args = parser.parse_args()

    train_data_public = pd.read_csv("data/train_public.csv")
    validation_data_public = pd.read_csv("data/validation_public.csv")
    test_data_public = pd.read_csv("data/train_internet.csv")

    dataset = Dataset(
        train_data_public=train_data_public,
        validation_data=validation_data_public,
        test_data=test_data_public,
    )

    dataset.preprocess()

    if args.model_type == "LR":
        model = logistic_regression.LogisticRegression(
            train_feature=dataset.train_feature,
            train_label=dataset.train_label,
            evaluate_feature=dataset.validation_feature,
            evaluate_label=dataset.validation_label,
        )
        prefix = "logistic_regression"
    elif args.model_type == "XGB":
        model = xgboost.XGBoost(
            train_feature=dataset.train_feature,
            train_label=dataset.train_label,
            evaluate_feature=dataset.validation_feature,
            evaluate_label=dataset.validation_label,
        )
        prefix = "xgboost"
    elif args.model_type == "NN":
        model = neural_network.NeuralNetwork(
            train_feature=dataset.train_feature,
            train_label=dataset.train_label,
            evaluate_feature=dataset.validation_feature,
            evaluate_label=dataset.validation_label,
        )
        prefix = "neural_network"
    elif args.model_type == "RF":
        model = random_forest.RandomForest(
            train_feature=dataset.train_feature,
            train_label=dataset.train_label,
            evaluate_feature=dataset.validation_feature,
            evaluate_label=dataset.validation_label,
        )
        prefix = "random_forest"

    model.train()
    print(f"{prefix} validation roc: {model.validate()}")
    pd.DataFrame(
        {
            "id": dataset.test_id,
            "isDefault": model.predict(dataset.test_feature),
        }
    ).to_csv(f"shit/{prefix}_submission.csv", index=False)

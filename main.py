import argparse
import pandas as pd
from data.pre_process import Dataset
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
    parser.add_argument("-I", "--internet", type=bool, default=False, required=False)
    args = parser.parse_args()

    train_data_public = pd.read_csv("data/train_public.csv")
    train_data_internet = (
        pd.read_csv("data/train_internet.csv") if args.internet else None
    )
    test_data_public = pd.read_csv("data/test_public.csv")

    dataset = Dataset(
        train_data_public=train_data_public,
        train_data_internet=train_data_internet,
        test_data=test_data_public,
    )

    dataset.preprocess()

    if args.model_type == "LR":
        model = logistic_regression.LogisticRegression(
            feature=dataset.train_feature, label=dataset.train_label
        )
        prefix = "logistic_regression"
    elif args.model_type == "XGB":
        model = xgboost.XGBoost(
            feature=dataset.train_feature, label=dataset.train_label
        )
        prefix = "xgboost"
    elif args.model_type == "NN":
        model = neural_network.NeuralNetwork(
            feature=dataset.train_feature, label=dataset.train_label
        )
        prefix = "neural_network"
    elif args.model_type == "RF":
        model = random_forest.RandomForest(
            feature=dataset.train_feature, label=dataset.train_label
        )
        prefix = "random_forest"

    model.train()
    print(f"{prefix} validation roc: {model.validate()}")
    pd.DataFrame(
        {
            "id": dataset.test_id,
            "isDefault": model.predict(dataset.test_feature),
        }
    ).to_csv(f"results/{prefix}_submission.csv", index=False)

import argparse
import pandas as pd
from data.pre_process import Dataset
from models.model import Model
from models import logistic_regression, support_vector_machine, neural_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-M",
        "--model_type",
        type=str,
        required=True,
        choices=["LR", "SVM", "NN"],
    )
    args = parser.parse_args()

    train_data_public = pd.read_csv("data/train_public.csv")
    train_data_internet = pd.read_csv("data/train_internet.csv")
    test_data_public = pd.read_csv("data/test_public.csv")

    dataset = Dataset(
        train_data_public=train_data_public,
        train_data_internet=train_data_internet,
        test_data_public=test_data_public,
    )

    dataset.preprocess()

    if args.model_type == "LR":
        model = logistic_regression.LogisticRegression(
            feature=dataset.train_feature, label=dataset.train_label
        )
        prefix = "logistic_regression"
    elif args.model_type == "SVM":
        model = support_vector_machine.SupportVectorMachine(
            feature=dataset.train_feature, label=dataset.train_label
        )
        prefix = "support_vector_machine"
    elif args.model_type == "NN":
        model = neuralNetwork = neural_network.NeuralNetwork(
            feature=dataset.train_feature, label=dataset.train_label
        )
        prefix = "neural_network"

    model.train()
    print(f"{prefix} validation roc: {model.validate()}")
    pd.DataFrame(
        {
            "id": dataset.test_id,
            "isDefault": model.predict(dataset.test_feature),
        }
    ).to_csv(f"{prefix}_submission.csv")

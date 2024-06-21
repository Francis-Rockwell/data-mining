import pandas as pd
from data.pre_process import Dataset
from models.model import Model
from models import logistic_regression, support_vector_machine, neural_network

if __name__ == "__main__":
    train_data_public = pd.read_csv("data/train_public.csv")
    train_data_internet = pd.read_csv("data/train_internet.csv")
    test_data_public = pd.read_csv("data/test_public.csv")

    dataset = Dataset(
        train_data_public=train_data_public,
        train_data_internet=train_data_internet,
        test_data_public=test_data_public,
    )

    dataset.preprocess()

    logisticRegression = logistic_regression.LogisticRegression(
        feature=dataset.train_feature, label=dataset.train_label
    )
    supportVectorMachine = support_vector_machine.SupportVectorMachine(
        feature=dataset.train_feature, label=dataset.train_label
    )
    neuralNetwork = neural_network.NeuralNetwork(
        feature=dataset.train_feature, label=dataset.train_label
    )

    # model_prefix = ["logistic_regression", "support_vector_machine", "neural_network"]
    model_prefix = ["logistic_regression", "support_vector_machine"]

    # models: list[Model] = [logisticRegression, supportVectorMachine, neuralNetwork]
    models: list[Model] = [logisticRegression, supportVectorMachine]

    for index, model in enumerate(models):
        model.train()
        print(f"{model_prefix[index]} validation roc: {model.validate()}")
        result = pd.DataFrame(
            {
                "id": dataset.test_id,
                "isDefault": model.predict(dataset.test_feature),
            }
        )
        result.to_csv(f"{model_prefix[index]}_submission.csv")

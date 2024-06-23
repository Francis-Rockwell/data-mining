import pandas as pd

num = 100000
true_internet = pd.read_csv("data/train_internet.csv")
predict_internet = pd.read_csv("shit/random_forest_submission.csv")

true_internet["diff"] = (
    true_internet["isDefault"] - predict_internet["isDefault"]
).abs()

true_internet = true_internet.sort_values(by=["diff"], ascending=True).head(num)

true_internet.to_csv("data/select_train_internet.csv")

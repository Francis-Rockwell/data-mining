from sklearn.model_selection import train_test_split
import pandas as pd

train_data_public = pd.read_csv("data/train_public.csv")

train_data, validation_data = train_test_split(
    train_data_public, test_size=0.2, random_state=42
)

train_data.to_csv("data/train_public_split.csv")
validation_data.to_csv("data/validation_public.csv")

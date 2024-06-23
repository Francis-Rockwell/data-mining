import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .encoder import WorkYearEncoder, IssueDateEncoder


class Dataset:
    def __init__(
        self,
        train_data_public: pd.DataFrame,
        validation_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ):
        self.train_data_public = train_data_public[
            train_data_public["isDefault"].notna()
        ]
        self.test_data = test_data
        self.validation_data = validation_data

        self.numerical_feature = [
            "total_loan",
            "year_of_loan",
            "interest",
            "monthly_payment",
            "debt_loan_ratio",
            "del_in_18month",
            "scoring_low",
            "scoring_high",
            "pub_dero_bankrup",
            "recircle_b",
            "recircle_u",
            "f0",
            "f1",
            "f2",
            "f3",
            "f4",
            "early_return",
            "early_return_amount",
            "early_return_amount_3mon",
        ]

        self.categorical_feature = [
            "class",
            "employer_type",
            "industry",
            "house_exist",
            "censor_status",
            "use",
            "post_code",
            "region",
            "initial_list_status",
            "earlies_credit_mon",
            "title",
        ]

        self.other_feature = ["work_year", "issue_date"]
        self.not_feature = [
            "loan_id",
            "user_id",
            "policy_code",
            "known_outstanding_loan",
            "known_dero",
            "app_type",
        ]

        self.train_data = self.train_data_public

    def preprocess(self):
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

        workyear_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("workyear", WorkYearEncoder()),
            ]
        )

        issuedate_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("issuedate", IssueDateEncoder()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numeric_transformer, self.numerical_feature),
                ("categorical", categorical_transformer, self.categorical_feature),
                ("workyear", workyear_transformer, ["work_year"]),
                ("issuedate", issuedate_transformer, ["issue_date"]),
            ]
        )

        self.train_feature = preprocessor.fit_transform(self.train_data)
        self.train_label = self.train_data["isDefault"]
        self.validation_feature = preprocessor.transform(self.validation_data)
        self.validation_label = self.validation_data["isDefault"]
        self.test_id: pd.DataFrame = self.test_data["loan_id"]
        self.test_feature = preprocessor.transform(self.test_data)

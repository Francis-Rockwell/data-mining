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
        train_data_internet: pd.DataFrame,
        test_data: pd.DataFrame,
    ):
        self.train_data_public = train_data_public[
            train_data_public["isDefault"].notna()
        ]
        self.train_data_internet = (
            None
            if train_data_internet is None
            else train_data_internet[train_data_internet["isDefault"].notna()]
        )
        self.test_data = test_data

        if self.train_data_internet is None:
            self.numerical_feature = [
                "total_loan",
                "year_of_loan",
                "interest",
                "monthly_payment",
                "debt_loan_ratio",
                "del_in_18month",
                "scoring_low",
                "scoring_high",
                "known_outstanding_loan",
                "known_dero",
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
                "app_type",
                "earlies_credit_mon",
                "title",
            ]

            self.other_feature = ["work_year", "issue_date"]
            self.not_feature = ["loan_id", "user_id", "policy_code"]

            self.train_data = self.train_data_public

        else:
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
                "user_id",
                "policy_code",
                "app_type",
            ]

            common_feature = (
                ["isDefault"]
                + self.numerical_feature
                + self.categorical_feature
                + self.other_feature
            )

            self.train_data_internet = self.train_data_internet[
                ~self.train_data_internet["loan_id"].isin(
                    set(self.train_data_public["loan_id"])
                )
            ]
            self.train_data = pd.merge(
                self.train_data_public[common_feature],
                self.train_data_internet[common_feature],
                on=common_feature,
                how="outer",
            )

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
        self.test_id: pd.DataFrame = self.test_data["loan_id"]
        self.test_feature = preprocessor.transform(self.test_data)

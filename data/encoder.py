from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class WorkYearEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):

        def mapping(workyear):
            workyear: str = workyear[0]
            if workyear == "10+ years":
                return 11
            elif workyear == "< 1 year":
                return 0
            else:
                return int(workyear.split()[0])

        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = np.copy(X)
        for idx in range(X.shape[0]):
            if pd.isnull(X[idx]):
                X_encoded[idx] = 0
            else:
                X_encoded[idx] = self.mapping(X[idx])
        return X_encoded.astype(int)


class IssueDateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):

        def mapping(issuedate):
            issuedate: str = issuedate[0]
            if "/" in issuedate:
                year, month, _ = issuedate.split("/")
            elif "-" in issuedate:
                year, month, _ = issuedate.split("-")
            else:
                raise NotImplementedError
            return [int(year), int(month)]

        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = np.copy(X)
        encoded_columns = []
        for value in X_encoded:
            if pd.isnull(value):
                encoded_value = [-1, -1, -1]
            else:
                encoded_value = self.mapping(value)
            encoded_columns.append(encoded_value)
        return np.array(encoded_columns).astype(int)

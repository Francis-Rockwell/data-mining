from .model import Model
import xgboost as xgb


class XGBoost(Model):
    def __init__(
        self, train_feature, train_label, validation_feature, validation_label
    ):
        super().__init__(
            train_feature, train_label, validation_feature, validation_label
        )
        self.model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]

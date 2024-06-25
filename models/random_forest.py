from .model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForest(Model):
    def __init__(
        self, train_feature, train_label, validation_feature, validation_label
    ):
        super().__init__(
            train_feature, train_label, validation_feature, validation_label
        )
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]

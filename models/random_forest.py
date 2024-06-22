from .model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):
    def __init__(self, feature, label, split=0.2):
        super().__init__(feature, label, split)
        self.model = RandomForestClassifier(n_estimators=5000, random_state=42)

    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]

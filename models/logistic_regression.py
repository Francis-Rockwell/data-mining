from .model import Model
import sklearn.linear_model


class LogisticRegression(Model):
    def __init__(self, feature, label, split=0.2):
        super().__init__(feature, label, split)
        self.model = sklearn.linear_model.LogisticRegression(max_iter=1000)

    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]

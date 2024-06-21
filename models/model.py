from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, feature, label, split=0.2):
        (
            self.train_feature,
            self.evaluate_feature,
            self.train_label,
            self.evaluate_label,
        ) = train_test_split(feature, label, test_size=split, random_state=42)

    def train(self):
        raise NotImplementedError

    def predict(self, feature):
        raise NotImplementedError

    def validate(self):
        return roc_auc_score(self.evaluate_label, self.predict(self.evaluate_feature))

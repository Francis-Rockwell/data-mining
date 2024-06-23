from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, train_feature, train_label, evaluate_feature, evaluate_label):
        self.train_feature = train_feature
        self.train_label = train_label
        self.evaluate_feature = evaluate_feature
        self.evaluate_label = evaluate_label

    def train(self):
        raise NotImplementedError

    def predict(self, feature):
        raise NotImplementedError

    def validate(self):
        return roc_auc_score(self.evaluate_label, self.predict(self.evaluate_feature))

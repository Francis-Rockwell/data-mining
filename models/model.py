from sklearn.metrics import roc_auc_score


class Model:
    def __init__(self, feature, label, split=0.8):
        self.split = split
        self.train_num = int(len(feature) * split)
        self.evaluate_num = len(feature) - self.train_num
        self.train_feature = feature[: self.train_num]
        self.train_label = label[: self.train_num]
        self.evaluate_feature = feature[self.train_num :]
        self.evaluate_label = label[self.train_num :]

    def train(self):
        pass

    def predict(self, feature):
        pass

    def evaluate(self):
        return roc_auc_score(self.evaluate_label, self.predict(self.evaluate_feature))

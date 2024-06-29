from sklearn.metrics import roc_auc_score


class Model:
    def __init__(
        self, train_feature, train_label, validation_feature, validation_label
    ):
        self.train_feature = train_feature
        self.train_label = train_label
        self.validation_feature = validation_feature
        self.validation_label = validation_label

    def train(self):
        raise NotImplementedError

    def predict(self, feature):
        raise NotImplementedError

    def validate(self):
        return roc_auc_score(
            self.validation_label, self.predict(self.validation_feature)
        )

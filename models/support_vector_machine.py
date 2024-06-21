from .model import Model
import sklearn.svm as svm


class SupportVectorMachine(Model):
    def __init__(self, feature, label, split=0.2):
        super().__init__(feature, label, split)
        self.model = svm.SVC(kernel="linear", probability=True)

    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]

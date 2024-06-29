from .model import Model
import sklearn.linear_model
import numpy as np
import tqdm


class LogisticRegression(Model):
    def __init__(
        self, train_feature, train_label, validation_feature, validation_label
    ):
        super().__init__(
            train_feature, train_label, validation_feature, validation_label
        )
        self.model = sklearn.linear_model.LogisticRegression(max_iter=10000)

    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def cost(predcit, y):
    return 0.5 * np.sum(np.square(predcit - y))


# class LogisticRegression(Model):
#     def __init__(
#         self, train_feature, train_label, validation_feature, validation_label
#     ):
#         super().__init__(
#             train_feature, train_label, validation_feature, validation_label
#         )
#         self.feature_num = self.train_feature.shape[1]
#         self.train_num = self.train_feature.shape[0]
#         self.validation_num = self.validation_feature.shape[1]
#         self.train_label = np.reshape(self.train_label, (-1, 1))
#         self.max_iter = 1000000
#         self.learning_rate = 0.0001
#         self.tol = 0.0001

#         self.weights = np.zeros((self.feature_num, 1))
#         self.bias = 0

#     def train(self):

#         final_cost = 0.0
#         for _ in tqdm.tqdm(range(self.max_iter)):
#             predcit = sigmoid(np.dot(self.train_feature, self.weights) + self.bias)

#             self.weights += self.learning_rate * (
#                 (1 / self.train_num)
#                 * np.dot(np.transpose(self.train_feature), (self.train_label - predcit))
#             )

#             self.bias += self.learning_rate * (
#                 (1 / self.train_num) * np.sum(self.train_label - predcit)
#             )

#             final_cost = cost(predcit, self.train_label)
#             if final_cost < self.tol:
#                 print("logistic regression converges")
#                 break
#         print(f"cost: {final_cost}")

#     def predict(self, feature):
#         return np.squeeze(sigmoid(np.dot(feature, self.weights)))

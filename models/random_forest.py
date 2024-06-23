from .model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForest(Model):
    def __init__(self, train_feature, train_label, evaluate_feature, evaluate_label):
        super().__init__(train_feature, train_label, evaluate_feature, evaluate_label)
        self.model = RandomForestClassifier(n_estimators=1000, random_state=42)
        # self.param_grid = {
        #     "n_estimators": [500, 1000, 2000],
        #     "max_depth": [None, 10, 20, 30],
        #     "min_samples_split": [2, 5, 10],
        #     "min_samples_leaf": [1, 2, 4],
        # }
        # self.grid_search = GridSearchCV(
        #     estimator=self.model,
        #     param_grid=self.param_grid,
        #     cv=5,
        #     scoring="accuracy",
        #     n_jobs=-1,
        # )

    def train(self):
        self.model.fit(self.train_feature, self.train_label)
        # self.grid_search.fit(self.train_feature, self.train_label)
        # self.best_model = self.grid_search.best_estimator_

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]

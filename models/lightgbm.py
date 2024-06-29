from .model import Model
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV


class LightGBM(Model):
    def __init__(
        self, train_feature, train_label, validation_feature, validation_label
    ):
        super().__init__(
            train_feature, train_label, validation_feature, validation_label
        )

        self.model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            n_estimators=500,
            random_state=42,
            verbose=0,
            n_jobs=8,
            num_leaves=16,
            max_depth=12,
            subsample=0.8,
            learning_rate=0.01,
            min_child_samples=30,
            colsample_bytree=0.8,
            # device = 'cuda'
        )

    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    # def train(self):
    # param_grid = {
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "min_child_samples": [30, 50, 70],
    #     "subsample": [0.8, 0.9, 1.0],
    #     "colsample_bytree": [0.8, 0.9, 1.0],
    # }

    # grid_search = GridSearchCV(
    #     estimator=self.model,
    #     param_grid=param_grid,
    #     scoring="roc_auc",
    #     cv=5,
    #     verbose=0,
    #     n_jobs=8,
    # )

    # grid_search.fit(self.train_feature, self.train_label)

    # self.model = grid_search.best_estimator_

    # print("Best parameters found: ", grid_search.best_params_)
    # print("Best AUC score: ", grid_search.best_score_)

    # self.model.fit(
    #     self.train_feature,
    #     self.train_label,
    #     eval_metric="auc",
    #     eval_set=[(self.validation_feature, self.validation_label)],
    # )

    def predict(self, feature):
        return self.model.predict_proba(feature)[:, 1]

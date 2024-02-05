from __future__ import annotations
from copy import deepcopy

import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

from . import goodness as good


def hasmeth(obj: object, name: str):
    if hasattr(obj, name):
        attr = getattr(obj, name)
        if callable(attr):
            return True
    return False


class MultiRegressor():

    def __init__(self, template_regressor) -> None:
        assert hasmeth(template_regressor, 'fit')
        assert hasmeth(template_regressor, 'predict')
        self.template_regressor = template_regressor

    def init_univariate_regressors(self) -> None:
        self.univariate_regressors = [
            deepcopy(self.template_regressor) for _ in range(self.num_y_vars)]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.num_y_vars = y.shape[0]
        self.init_univariate_regressors()

        x = x.T
        y = y.T

        self.scaler_x = preprocessing.StandardScaler()
        self.scaler_y = preprocessing.StandardScaler()

        self.scaler_x.fit(x)
        self.scaler_y.fit(y)

        x = self.scaler_x.transform(x)
        y = self.scaler_y.transform(y)

        for idx_y_var in range(self.num_y_vars):
            self.univariate_regressors[idx_y_var].fit(x, y[:, idx_y_var])

    def predict(self, x: np.ndarray) -> np.ndarray:

        x = x.T
        x = self.scaler_x.transform(x)

        y = [
            self.univariate_regressors[idx_y_var].predict(x)
            for idx_y_var in range(self.num_y_vars)
        ]
        y = np.stack(y, axis=1)
        y = self.scaler_y.inverse_transform(y)
        y = y.T

        return y


def train_multiregressor(
    xtrain: np.ndarray[np.float32],
    ytrain: np.ndarray[np.float32],
    downsamp: np.uint32,
    alpha: float,
) -> linear_model.Lasso:

    xtrain = xtrain[:, ::downsamp]
    ytrain = ytrain[:, ::downsamp]

    template_regressor = linear_model.Lasso(
        alpha=alpha,
        fit_intercept=False,
        precompute=False,
        positive=False,
        tol=0.1,  # much larger than default, which is 1e-4
        max_iter=1000,
    )
    multiregressor = MultiRegressor(template_regressor)
    multiregressor.fit(xtrain, ytrain)

    return multiregressor


def evaluate_multiregressor(
    multiregressor: MultiRegressor,
    xinfer: np.ndarray[np.float32],
    yinfer: np.ndarray[np.float32],
    downsamp: np.uint32,
) -> dict:

    xinfer = xinfer[:, ::downsamp]
    yinfer = yinfer[:, ::downsamp]

    yinfer_pred_raw = multiregressor.predict(xinfer)

    metrics_infer = good.compute_regression_metrics(yinfer, yinfer_pred_raw)

    return metrics_infer


def main():
    pass


if __name__ == '__main__':
    main()

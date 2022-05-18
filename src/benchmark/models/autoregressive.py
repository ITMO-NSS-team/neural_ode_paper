import warnings

import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

from benchmark.models.common import Model

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)


class AutoRegressiveModel(Model):
    def __init__(self, out_steps, order=(0, 0, 0), seasonal_order=(1, 0, 1, 24)) -> None:
        super().__init__()
        self.out_steps = out_steps
        self.order = order
        self.seasonal_order = seasonal_order

    def __call__(self, *args, **kwargs):
        preds = self.predict_for_x(*args)
        return np.expand_dims(np.array(preds), axis=-1)

    def compile_and_train(self, gen):
        pass

    def evaluate(self, data):
        X_eval, y_eval = data
        y_pred = self.predict_for_x(X_eval)
        return mean_absolute_error(y_eval.reshape(y_eval.shape[0], y_eval.shape[1]), np.array(y_pred))

    def predict_for_x(self, X_eval):
        y_pred = []
        for X_i in tqdm(X_eval):
            try:
                model = ARIMA(X_i, order=self.order, seasonal_order=self.seasonal_order)
                model_fit = model.fit()
                y_pred_i = model_fit.forecast(steps=self.out_steps)
                y_pred.append(y_pred_i)
            except:
                y_pred.append(np.resize(X_i.reshape(-1), self.out_steps))
        return y_pred

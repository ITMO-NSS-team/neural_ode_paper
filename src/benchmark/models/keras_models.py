import tensorflow as tf

from benchmark.models.common import Model
from benchmark.utils import compile_and_fit


# Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series
class RepeatModel(Model):
    def __init__(self):
        class RepeatBaseline(tf.keras.Model):
            def call(self, inputs, training=None, mask=None):
                return inputs

        self._repeat_baseline = RepeatBaseline()

    def __call__(self, *args, **kwargs):
        return self._repeat_baseline(*args)

    def compile_and_train(self, gen):
        self._repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                                      metrics=[tf.metrics.MeanAbsoluteError()])

    def evaluate(self, data):
        X_eval, y_eval = data
        return self._repeat_baseline.evaluate(X_eval, y_eval, steps=1)[1]

    def reset(self):
        pass


class DenseModel(Model):
    def __init__(self, out_steps, batch_size, epochs, hidden_size=128):
        self.multi_linear_model = self.create_model(hidden_size, out_steps)
        self.hidden_size = hidden_size
        self.out_steps = out_steps
        self.batch_size = batch_size
        self.epochs = epochs

    def create_model(self, hidden_size, out_steps):
        return tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(out_steps),
            tf.keras.layers.Reshape([out_steps, 1])
        ])

    def __call__(self, *args, **kwargs):
        return self.multi_linear_model(*args)

    def compile_and_train(self, gen):
        compile_and_fit(self.multi_linear_model, gen, self.batch_size, self.epochs)

    def evaluate(self, data):
        X_eval, y_eval = data
        return self.multi_linear_model.evaluate(X_eval, y_eval, steps=1)[1]

    def reset(self):
        self.multi_linear_model = self.create_model(self.hidden_size, self.out_steps)


class LSTMModel(Model):
    def __init__(self, out_steps, batch_size, epochs, hidden_size_lstm=32, hidden_size_dense=128):
        self.multi_lstm_model = self.create_model(hidden_size_dense, hidden_size_lstm, out_steps)
        self.out_steps = out_steps
        self.hidden_size_lstm = hidden_size_lstm
        self.hidden_size_dense = hidden_size_dense
        self.batch_size = batch_size
        self.epochs = epochs

    def create_model(self, hidden_size_dense, hidden_size_lstm, out_steps):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_size_lstm, return_sequences=False),
            tf.keras.layers.Dense(hidden_size_dense),
            tf.keras.layers.Dense(out_steps,
                                  kernel_initializer=tf.keras.initializers.zeros()),
            tf.keras.layers.Reshape([out_steps, 1])
        ])

    def __call__(self, *args, **kwargs):
        return self.multi_lstm_model(*args)

    def compile_and_train(self, gen):
        compile_and_fit(self.multi_lstm_model, gen, self.batch_size, self.epochs)
        self.multi_lstm_model.save('test')

    def evaluate(self, data):
        X_eval, y_eval = data
        return self.multi_lstm_model.evaluate(X_eval, y_eval, steps=1)[1]

    def reset(self):
        self.multi_lstm_model = self.create_model(self.hidden_size_dense, self.hidden_size_lstm, self.out_steps)

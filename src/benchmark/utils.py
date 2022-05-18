import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tqdm import tqdm


def get_dataset(dataset='temp', noise_level=0):
    if dataset == 'temp':
        df = load_temp_dataset()
    elif dataset == 'venice':
        df = pd.read_csv('data/venezia.csv')
        df = df[['level']]
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    n = len(df)
    train_ratio = 0.7
    val_ratio = 0.2

    df_mean = df.mean()
    df_std = df.std()

    df = (df - df_mean) / df_std

    df += np.random.normal(0, float(noise_level), size=df.shape)

    train_df = df[:int(n * train_ratio)]
    val_df = df[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
    test_df = df[int(n * (train_ratio + val_ratio)):]

    return train_df, val_df, test_df


def load_temp_dataset():
    dfs = []
    key = None
    for i in tqdm(range(2011, 2021, 1)):
        df = pd.read_csv(f'data/hourly_TEMP_{i}.csv', engine='c')
        df['Latitude'] = round(df['Latitude'], 4)
        df['Longitude'] = round(df['Longitude'], 4)
        groupby = df[df['State Name'] == 'Massachusetts'].groupby(['Latitude', 'Longitude'])
        if key is None:
            groups = groupby.groups
            keys = list(groups.keys())
            key = keys[1]
        df = groupby.get_group(key)
        df = df[['Sample Measurement']]
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def compile_and_fit(model: Model, window, batch_size, epochs):
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    X_train, y_train = window.train
    X_val, y_val = window.val
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_val, y_val))
    return history


# Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series
class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 out_dir, label_columns=None, ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.out_dir = out_dir

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def plot(self, plot_col=None, model=None, max_subplots=3):
        if plot_col is None:
            plot_col = self.label_columns[0]
        inputs, labels = self.example
        inputs = inputs[::-1]
        labels = labels[::-1]
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        fig, axes2d = plt.subplots(nrows=3, figsize=(12, 8))
        if model is not None:
            predictions = model(inputs[:max_n])

        for n in range(max_n):
            axes2d[n].plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            axes2d[n].scatter(self.label_indices, labels[n, :, label_col_index],
                              edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                axes2d[n].scatter(self.label_indices, predictions[n, :, label_col_index],
                                  marker='X', edgecolors='k', label='Predictions',
                                  c='#ff7f0e', s=64)
        font = {'size': 16}
        matplotlib.rc('font', **font)
        fig.text(0.0125, 0.5, 'Water level, cm (normed)', va='center', ha='center', rotation='vertical')
        fig.text(0.5, 0.015, 'Time, hours', va='center', ha='center')
        plt.tight_layout()
        plt.legend(loc='lower left')
        fig_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
        plt.savefig(os.path.join(self.out_dir, f'{fig_datetime}.png'))
        plt.show()

    def make_dataset(self, data):
        data = np.array(data)
        ds_input = []
        ds_labels = []
        for i in range(0, len(data), self.input_width + self.label_width):
            ds_input.append(data[i:i + self.input_width])
            ds_labels.append(data[i + self.input_width:i + self.input_width + self.label_width])
        ds_input = np.array(ds_input[:-1])
        ds_labels = np.array(ds_labels[:-1])
        return ds_input, ds_labels

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        return self.train


def save_results(out_path, model_scores, agg):
    # noinspection PyTypeChecker
    model_scores.to_csv(f'{out_path}/results.csv', index=False)
    agg.to_csv(f'{out_path}/agg.csv')

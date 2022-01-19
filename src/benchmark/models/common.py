from abc import ABC, abstractmethod
from itertools import cycle

import torch


class Model(ABC):
    @abstractmethod
    def compile_and_train(self, gen):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass

    @abstractmethod
    def reset(self):
        pass


def batch_generator(ds, batch_size, use_torch=False):
    ds_input, ds_labels = ds
    ds_input = ds_input[:ds_input.shape[0] - ds_input.shape[0] % batch_size]
    ds_labels = ds_labels[:ds_labels.shape[0] - ds_labels.shape[0] % batch_size]
    if use_torch:
        return cycle(
            zip(torch.from_numpy(ds_input).type(torch.float32), torch.from_numpy(ds_labels).type(torch.float32)))
    else:
        ds_input = ds_input.reshape((-1, batch_size, ds_input.shape[1], 1))
        ds_labels = ds_labels.reshape((-1, batch_size, ds_labels.shape[1], 1))
        return cycle(zip(ds_input, ds_labels))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn, optim
from torch import sin, cos

from benchmark.models.common import Model, batch_generator


class ClosedFormModel(Model, nn.Module):
    def __init__(self, out_steps, size, hidden_size, batch_size, n_epochs=30, n_iters=300, integration_limit=4 * np.pi,
                 coef_loss_threshold=0.1, approx_steps=100):
        super().__init__()
        # General params
        self.out_steps = out_steps
        self.size = size
        self.loss_function = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.coef_loss_threshold = coef_loss_threshold
        self.hidden_size = hidden_size
        self.size = size
        self.approx_steps = approx_steps
        self.coefs, self.coef_loss_threshold, self.t_inference, self.C_inference, self.t_converter, self.C_converter, \
        self.refinement_lstm, self.refinement_fcnn, self.t_0, self.C = self.initialize_trainable_fields(
            coef_loss_threshold, hidden_size, size)

        self.integration_limit = integration_limit

    def initialize_trainable_fields(self, coef_loss_threshold, hidden_size, size):
        coefs = nn.Parameter(5 * torch.rand(size, 2), requires_grad=False)
        coef_loss_threshold = coef_loss_threshold
        t_inference = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        C_inference = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        t_converter = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                                    nn.Linear(hidden_size, 1, bias=True))
        C_converter = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
                                    nn.Linear(hidden_size, 2 * size + 1, bias=True))
        refinement_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        refinement_fcnn = nn.Linear(hidden_size, 1, bias=True)
        t_0 = nn.Parameter(torch.zeros(1))
        C = nn.Parameter(torch.rand(2 * size + 1) - 1 / 2)
        return coefs, coef_loss_threshold, t_inference, C_inference, t_converter, C_converter, refinement_lstm, \
               refinement_fcnn, t_0, C

    def __call__(self, *args, **kwargs):
        data = args
        data = torch.tensor(data).type(torch.float32)
        out = []
        for input_part in data[0]:
            t = self.get_t()
            out.append(self.forward_nn(t, input_part))

        pred = torch.unsqueeze(torch.stack(out), dim=-1)
        return pred.detach()

    def forward_nn(self, t, y):
        C, t_0 = self.get_initial_conditions(y)
        output = self.forward(t, self.t_0 + t_0, self.C + C)
        output = output.view((1, output.shape[0], 1))
        output, _ = self.refinement_lstm(output)
        output = self.refinement_fcnn(output)
        return output.reshape(-1)

    def get_initial_conditions(self, y):
        y_2d = torch.unsqueeze(y, dim=0)
        _, (t_out, _) = self.t_inference(y_2d)
        _, (C_out, _) = self.C_inference(y_2d)
        t_0 = self.t_converter(torch.squeeze(t_out))
        C = self.C_converter(torch.squeeze(C_out))
        return C, t_0

    def forward(self, t, t_0, C, n_steps=None):
        if n_steps is None:
            n_steps = t.shape[0]
        out = torch.zeros_like(t)
        t = torch.zeros_like(t)
        step = self.get_t(n_steps).max() / n_steps
        for i in range(0, n_steps):
            t[i] += (t_0 + i * step)[0]
        for i in range(1, self.size + 1):
            out += C[2 * i - 1] * cos(self.coefs[i - 1] * t) + C[2 * i] * sin(self.coefs[i - 1] * t)
        out += C[0]
        return out

    def get_t(self, steps=None):
        if steps is None:
            steps = self.out_steps
        return torch.linspace(0., 4 * np.pi * steps / self.approx_steps, self.out_steps, requires_grad=False)

    def compile_and_train(self, gen):
        train_gen = batch_generator(gen.train, self.batch_size, use_torch=True)
        current_loss = None
        itr = 0
        while current_loss is None or current_loss > self.coef_loss_threshold:
            itr += 1
            train_batch = next(train_gen)
            input_part, label_part = train_batch
            t = self.get_t()
            with torch.no_grad():
                res = minimize(self.loss_fn, x0=self.get_initial_params(), args=[t, input_part, label_part],
                               method='COBYLA', options={'maxiter': 10000}, tol=1e-8)
            current_loss = res['fun']
            print(f'Iteration {itr}, loss: {current_loss}')
            plt.plot(self.forward_fixed(t, input_part).detach())
            plt.plot(label_part)
            plt.show()

        optimizer = optim.Adam(self.parameters())
        for epoch_id in range(1, self.n_epochs):
            losses = []
            for _ in range(1, self.n_iters):
                optimizer.zero_grad()
                train_batch = next(train_gen)
                input_part, label_part = train_batch
                t = self.get_t()
                train_loss = self.loss_function(self.forward_nn(t, input_part), label_part.reshape(-1))
                train_loss.backward()
                optimizer.step()
                losses.append(train_loss.item())
            print('Epoch {:04d} | Mean loss {:.6f}'.format(epoch_id, np.mean(losses).item()))

    def get_initial_params(self, max_amp=5):
        return np.concatenate((np.zeros(2), np.ones(2 * self.size), max_amp * np.random.sample(self.size)))

    def loss_fn(self, vec, args):
        t, input_part, label_part = args
        t_0, C, coefs = vec[:1], vec[1:2 * (self.size + 1)], vec[2 * (self.size + 1):3 * (self.size + 1) - 1]
        self.t_0 = nn.Parameter(torch.from_numpy(t_0))
        self.C = nn.Parameter(torch.from_numpy(C))
        self.coefs = nn.Parameter(torch.from_numpy(coefs))
        pred = self.forward_fixed(t, input_part).detach()[:self.approx_steps]
        label = label_part.reshape(-1)[:self.approx_steps]
        return self.loss_function(pred, label)

    def forward_fixed(self, t, y):
        return self.forward(t, self.t_0, self.C, self.approx_steps)

    def evaluate(self, data):
        losses = []
        for input_part, label_part in zip(*data):
            input_part, label_part = torch.tensor(input_part).type(torch.float32), \
                                     torch.tensor(label_part).type(torch.float32)
            t = self.get_t()
            pred = self.forward_nn(t, input_part)
            loss = torch.abs(label_part.reshape(-1) - pred).mean().item()
            losses.append(loss)
        return np.mean(losses)

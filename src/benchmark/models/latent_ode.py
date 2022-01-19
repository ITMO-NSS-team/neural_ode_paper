import logging
import os
from datetime import datetime
from itertools import cycle

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.distributions import Normal

from benchmark.models.common import Model, batch_generator
from lib import utils
from lib.create_latent_ode_model import create_LatentODE_model

logger = logging.getLogger('latent_ode')
logger.setLevel(logging.DEBUG)


class NeuralOdeModel(Model):
    def __init__(self, batch_size, num_batches=10000, lr=2e-3, ):
        self.device = torch.device('cuda:0')
        self.args, self.model = self.create_latent_ode_model()
        self.experimentID = str(datetime.now())
        self.lr = lr
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.img_path = None  # TODO
        self.viz = None  # TODO

    def create_latent_ode_model(self, input_dim=1, obsrv_std=0.01, z0_mean=0.0, z0_std=1.):
        class Args:
            def __init__(self, latents=30, poisson=False, gen_layers=1, units=100, rec_dims=20, rec_layers=1,
                         gru_units=100, z0_encoder='odernn', classif=False, linear_classif=False, dataset=None):
                self.latents = latents
                self.poisson = poisson
                self.gen_layers = gen_layers
                self.units = units
                self.rec_dims = rec_dims
                self.rec_layers = rec_layers
                self.gru_units = gru_units
                self.z0_encoder = z0_encoder
                self.classif = classif
                self.linear_classif = linear_classif
                self.dataset = dataset

        args = Args()
        input_dim = input_dim
        obsrv_std = torch.Tensor([obsrv_std]).to(self.device)
        z0_prior = Normal(torch.Tensor([z0_mean]).to(self.device), torch.Tensor([z0_std]).to(self.device))
        model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, self.device)
        return args, model

    def compile_and_train(self, gen):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr)
        ckpt_path = self.get_checkpoint()
        self.model.train()
        train_gen = batch_generator(gen.train, self.batch_size)
        val_gen = batch_generator(gen.val, self.batch_size)
        test_batch = next(val_gen)
        test_dict = self.to_batch_dict(test_batch)
        wait_until_kl_inc = 10

        for itr in range(1, self.num_batches):
            optimizer.zero_grad()
            utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=self.lr / 10)
            kl_coef = self.update_kl_coef(itr, wait_until_kl_inc)

            batch_dict = self.to_batch_dict(next(train_gen))
            train_res = self.model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
            train_res["loss"].backward()
            optimizer.step()

            if itr % 50 == 0:
                with torch.no_grad():
                    self.print_losses(itr, kl_coef, train_res, val_gen)
                    self.visualize_predictions(itr, test_dict)
        torch.save({'state_dict': self.model.state_dict()}, ckpt_path)

    def update_kl_coef(self, itr, wait_until_kl_inc):
        if itr // self.num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1 - 0.99 ** (itr // self.num_batches - wait_until_kl_inc))
        return kl_coef

    def print_losses(self, itr, kl_coef, train_res, val_gen):
        gen = cycle(map(lambda x: self.to_batch_dict(x), val_gen))
        test_res = utils.compute_loss_all_batches(self.model, gen, self.args, n_batches=5,
                                                  experimentID=self.experimentID, device=self.device,
                                                  n_traj_samples=self.batch_size, kl_coef=kl_coef)
        now = datetime.now()
        message = f'''{now}: Epoch {itr:04d} [Test seq (cond on sampled tp)]
                              | Loss {test_res["loss"].detach():.6f} | Likelihood {test_res["likelihood"].detach():.6f}
                              | KL fp {test_res["kl_first_p"]:.4f} | FP STD {test_res["std_first_p"]:.4f}|'''
        logger.info(f'Experiment {self.experimentID}')
        logger.info(message)
        logger.info(f'KL coef: {kl_coef}')
        logger.info(f'Train loss (one batch): {train_res["loss"].detach()}')
        logger.info(f'Train CE loss (one batch): {train_res["ce_loss"].detach()}')
        logger.info(f'Classification AUC (TEST): {test_res["auc"]:.4f}')
        logger.info(f'Test MSE: {test_res["mse"]:.4f}')
        logger.info(f'Classification accuracy (TRAIN): {train_res["accuracy"]:.4f}')
        logger.info(f'Classification accuracy (TEST): {test_res["accuracy"]:.4f}')
        logger.info(f'Poisson likelihood: {test_res["pois_likelihood"]}')
        logger.info(f'CE loss: {test_res["ce_loss"]}')

    def visualize_predictions(self, itr, test_dict):
        now = datetime.now()
        print(f"{now}: plotting....")
        plot_id = itr // 10
        self.viz.draw_all_plots_one_dim(test_dict, self.model, f'{self.img_path}_{self.experimentID}_{plot_id:03d}.png',
                                        self.experimentID, save=True)
        now = datetime.now()
        print(f"{now}: finished plotting")

    def get_checkpoint(self):
        ckpt_path = os.path.join('experiments/', "experiment_" + str(self.experimentID) + '.ckpt')
        utils.get_ckpt_model(ckpt_path, self.model, self.device)
        return ckpt_path

    def to_batch_dict(self, batch):
        batch_input, batch_labels = batch
        batch_dict = {
            'observed_data': torch.from_numpy(batch_input.copy()).type('float32').to(self.device),
            'observed_tp': torch.from_numpy(np.linspace(0, 0.5, batch_input.shape[1])).type('float32').to(self.device),
            'data_to_predict': torch.from_numpy(batch_labels.copy()).type('float32').to(self.device),
            'tp_to_predict': torch.from_numpy(np.linspace(0.5, 3.0, batch_labels.shape[1])).type('float32').to(
                self.device),
            'observed_mask': torch.from_numpy(np.ones_like(batch_input)).type('float32').to(self.device),
            'mask_predicted_data': None,
            'labels': None,
            'mode': 'extrap'
        }
        return batch_dict

    def __call__(self, *args, **kwargs):
        pred_y = self.predict_neural(*args)
        return pred_y.cpu().detach().numpy()[0]

    def predict_neural(self, args):
        gen = batch_generator(args, self.batch_size)
        batch_dict = self.to_batch_dict(next(gen))
        pred_y, info = self.model.get_reconstruction(batch_dict["tp_to_predict"], batch_dict["observed_data"],
                                                     batch_dict["observed_tp"], mask=batch_dict["observed_mask"],
                                                     n_traj_samples=1, mode=batch_dict["mode"])
        return pred_y

    def evaluate(self, data):
        _, y_eval = data
        pred_y = self.predict_neural(data)
        pred_y = pred_y.cpu().detach().numpy()[0]
        return mean_squared_error(y_eval.reshape(self.batch_size, -1), pred_y.reshape(self.batch_size, -1))

    def reset(self):
        pass

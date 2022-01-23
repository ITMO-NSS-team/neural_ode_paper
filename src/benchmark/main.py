import argparse
import logging

import numpy as np

from benchmark.models.autoregressive import AutoRegressiveModel
from benchmark.models.closed_form import ClosedFormModel
from benchmark.models.keras_models import RepeatModel, DenseModel, LSTMModel
from benchmark.models.latent_ode import NeuralOdeModel
from benchmark.utils import WindowGenerator, get_dataset

logging.basicConfig()
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser('Model benchmark')
parser.add_argument('--n-rep', type=int, default=5, help='Number of experiments to repeat with each model')
parser.add_argument('--dataset', type=str, default='venice', help='Dataset to load. Available: venice, temp')
parser.add_argument('--models', nargs='+', help='List of models to use',
                    default=['repeater', 'autoreg', 'linear', 'lstm', 'latent_ode', 'closed_form'])
parser.add_argument('--max-epochs', type=int, default=80, help='Number of epochs to fit keras models')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for fitting keras models')
parser.add_argument('--in-steps', type=int, default=100, help='Number of input points for each model')
parser.add_argument('--out-steps', type=int, default=100, help='Number of points to predict for each model')
parser.add_argument('--n-closed-form-components', type=int, default=2,
                    help='Number of periodic components for closed form model')
parser.add_argument('--closed-form-hidden-size', type=int, default=32,
                    help='Number of points to predict for each model')

if __name__ == '__main__':
    args = parser.parse_args()

    label_columns = ['level'] if args.dataset == 'venice' else ['Sample Measurement']
    df_train, df_test, df_val = get_dataset(dataset=args.dataset)
    multi_window = WindowGenerator(input_width=args.in_steps, label_width=args.out_steps, shift=args.out_steps,
                                   train_df=df_train, test_df=df_test, val_df=df_val, label_columns=label_columns)

    multi_window.plot()
    all_models = {
        'repeater': RepeatModel(),
        'linear': DenseModel(args.out_steps, args.batch_size, args.max_epochs),
        'lstm': LSTMModel(args.out_steps, args.batch_size, args.max_epochs),
        'autoreg': AutoRegressiveModel(args.out_steps),
        'latent_ode': NeuralOdeModel(args.batch_size, args.out_steps),
        'closed_form': ClosedFormModel(args.out_steps, args.n_closed_form_components, args.closed_form_hidden_size,
                                       args.batch_size)
    }

    models_to_run = [(name, model) for name, model in all_models.items() if name in args.models]

    multi_val_performance = {}
    multi_performance = {}

    for name, model in models_to_run:
        logger.info(f'Started benchmarking model {name}')
        model_val_results = []
        model_test_results = []
        for i in range(args.n_rep):
            model.compile_and_train(multi_window)
            val_res = model.evaluate(multi_window.val)
            test_res = model.evaluate(multi_window.test)
            model_val_results.append(val_res)
            model_test_results.append(test_res)
            multi_window.plot(model=model)
            model.reset()
            logger.info(f'Model: {name}, val results: {val_res:0.4f}, test results: {test_res:0.4f}')
        multi_val_performance[name] = (np.mean(model_val_results), np.std(model_val_results))
        multi_performance[name] = (np.mean(model_test_results), np.std(model_test_results))

    for name, (mean, std) in multi_performance.items():
        logger.info(f'Model: {name}, test_results: {mean:0.4f} +- {std:0.4f}')

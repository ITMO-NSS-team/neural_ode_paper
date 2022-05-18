import argparse
import logging
import os
from datetime import datetime

import pandas as pd

from benchmark.models.autoregressive import AutoRegressiveModel
from benchmark.models.closed_form import ClosedFormModel
from benchmark.models.keras_models import RepeatModel, DenseModel, LSTMModel
from benchmark.models.latent_ode import NeuralOdeModel
from benchmark.utils import WindowGenerator, get_dataset, save_results

logging.basicConfig()
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser('Model benchmark')
parser.add_argument('--n-rep', type=int, default=5, help='Number of experiments to repeat with each model')
parser.add_argument('--dataset', type=str, default='venice', help='Dataset to load. Available: venice, temp')
parser.add_argument('--models', nargs='+', help='List of models to use',
                    default=['repeater', 'autoreg', 'fc', 'lstm', 'latent_ode', 'closed_form'])
parser.add_argument('--max-epochs', type=int, default=80, help='Number of epochs to fit keras models')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for fitting keras models')
parser.add_argument('--in-steps', type=int, default=100, help='Number of input points for each model')
parser.add_argument('--out-steps', type=int, default=100, help='Number of points to predict for each model')
parser.add_argument('--n-closed-form-components', type=int, default=2,
                    help='Number of periodic components for closed form model')
parser.add_argument('--closed-form-hidden-size', type=int, default=32,
                    help='Number of points to predict for each model')

parser.add_argument('--noise-levels', nargs='+', default=[0.0],
                    help='Standard deviation of the noise added to scaled data')
parser.add_argument('--out-dir', type=str, default='out/',
                    help='Output directory for experiment results')
parser.add_argument('--skip-validation', action='store_true',
                    help='Whether to perform evaluation on validation dataset')
parser.add_argument('--num-batches', type=int, default=9000, help='Number of batches to train latent ODE model')
parser.add_argument('--ckpt-name', type=str, default=None, help='Name of checkpoint to load for latent ODE model')

if __name__ == '__main__':
    args = parser.parse_args()
    experiment_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    dir_name = f'{experiment_datetime}-{args.models}-noise-{args.noise_levels}-{args.dataset}-{args.out_steps}'
    out_path = os.path.join(args.out_dir, dir_name)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    label_columns = ['level'] if args.dataset == 'venice' else ['Sample Measurement']
    all_models = {
        'repeater': lambda: RepeatModel(),
        'fc': lambda: DenseModel(args.out_steps, args.batch_size, args.max_epochs),
        'lstm': lambda: LSTMModel(args.out_steps, args.batch_size, args.max_epochs),
        'autoreg': lambda: AutoRegressiveModel(args.out_steps),
        'latent_ode': lambda: NeuralOdeModel(args.batch_size, args.out_steps, args.num_batches, args.ckpt_name),
        'closed_form': lambda: ClosedFormModel(args.out_steps, args.n_closed_form_components,
                                               args.closed_form_hidden_size, args.batch_size)
    }

    models_to_run = [(name, model_method) for name, model_method in all_models.items() if name in args.models]

    model_scores = []
    for noise_level in args.noise_levels:
        df_train, df_val, df_test = get_dataset(dataset=args.dataset, noise_level=noise_level)
        multi_window = WindowGenerator(input_width=args.in_steps, label_width=args.out_steps, shift=args.out_steps,
                                       train_df=df_train, test_df=df_test, val_df=df_val, out_dir=out_path,
                                       label_columns=label_columns)

        multi_window.plot()
        logger.info(f'Started experiment with noise level {noise_level}')
        for name, model_method in models_to_run:
            logger.info(f'Started benchmarking model {name}')
            for i in range(args.n_rep):
                model = model_method()
                model.compile_and_train(multi_window)
                multi_window.plot(model=model)
                test_res = model.evaluate(multi_window.test)
                if args.skip_validation:
                    model_scores.append([name, noise_level, test_res])
                    logger.info(f'Model: {name}, test mae: {test_res:0.4f}')
                else:
                    val_res = model.evaluate(multi_window.val)
                    model_scores.append([name, noise_level, val_res, test_res])
                    logger.info(f'Model: {name}, val mae: {val_res:0.4f}, test mae: {test_res:0.4f}')

    if args.skip_validation:
        model_scores = pd.DataFrame(data=model_scores, columns=['model', 'noise_level', 'test_mae'])
        agg_dict = {'test_mae': ['mean', 'std']}
    else:
        model_scores = pd.DataFrame(data=model_scores, columns=['model', 'noise_level', 'val_mae', 'test_mae'])
        agg_dict = {'val_mae': ['mean', 'std'], 'test_mae': ['mean', 'std']}

    print('=' * 100)
    groups = model_scores.groupby(['model', 'noise_level'])
    agg = groups.agg(agg_dict)
    print(agg)
    print('=' * 100)
    print(model_scores)
    save_results(out_path, model_scores, agg)

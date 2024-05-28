import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import time
import argparse
import pandas as pd
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import TensorBoardLogger
import omegaconf
from pytorch_lightning.loggers import MLFlowLogger
import importlib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.utils import set_seed, AlphaRise
from src.utils.utils import get_checkpoint_filename, evaluate, get_absolute_path, log_data_seed, clear_tfevents, add_float_treatment, repeat_static, to_float, count_parameters, to_double
import warnings
from hydra.utils import get_original_cwd
import pickle

warnings.filterwarnings("ignore", category=UserWarning, module='fvcore.*')

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


@hydra.main(config_name=f'config.yaml', config_path='../configs/')
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    print('\n' + OmegaConf.to_yaml(args, resolve=True))
    set_seed(args['exp']['seed'])
    # set checkpoint
    current_date = time.strftime("%Y-%m-%d", time.localtime())
    model_name = args['model']['name']
    current_dir = get_absolute_path('./')
    print(current_dir)
    # exit()
    # get current time as pytorch-lightning version
    version = time.strftime("%H-%M-%S", time.localtime())
    
    original_cwd = get_original_cwd()
    args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
    path = os.path.join(args['exp']['processed_data_dir'], f"seed_{args['exp']['seed']}.pkl")
    if 'data_seed' in args.dataset:
        original_cwd = get_original_cwd()
        args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
        path = os.path.join(args['exp']['processed_data_dir'], f"seed_10.pkl")
        if not os.path.exists(path):
            print(f'{path} does not exist. Creating it now.')
            os.makedirs(args['exp']['processed_data_dir'], exist_ok=True)
            assert args.exp.seed == 10
            dataset_collection = instantiate(args.dataset, _recursive_=True)
            dataset_collection.process_data_multi()
            dataset_collection = to_float(dataset_collection)
            with open(path, 'wb') as file:
                pickle.dump(dataset_collection, file)
        else:
            with open(path, 'rb') as file:
                dataset_collection = pickle.load(file)
    else:
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        dataset_collection.process_data_multi()
        dataset_collection = to_float(dataset_collection)

    if args['dataset']['static_size'] > 0:
        # check if the dim of static features equals to 2
        dims = len(dataset_collection.train_f.data['static_features'].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    append_results = {}
    append_results['mean_y'] = dataset_collection.train_f.data['outputs'].mean()

    # set early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args['exp']['patience'],
        mode='min')
    # set learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    class_path = args["model"]["_target_"]
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    model = cls(dataset_collection, args)
    count_parameters(model, logger=logger) 
    # set Logger for pytorch-lightning
    logger_board = TensorBoardLogger(
        save_dir=current_dir, 
        name='', 
        version='')
    if args['exp']['logging']:
        mlf_logger = MLFlowLogger(
            experiment_name=f'{args["model"]["name"]}/{args["exp"]["mode"]}/{args["exp"]["name"]}',
            tracking_uri="http://localhost:5000",
            # save_dir = get_absolute_path('mlruns'),
        )
    else:
        mlf_logger = None
    if args.exp.logging:
        mlf_logger.log_metrics({'ymean': dataset_collection.train_f.data['outputs'].mean()})
        experiment_id = mlf_logger.experiment_id
        run_id = mlf_logger.run_id
        dirpath = os.path.join('checkpoints/', experiment_id, run_id)
    else:
        dirpath = None
    # set checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        mode='min'
    )
    # set trainer
    alpharise = AlphaRise()
    trainer = pl.Trainer(
        logger=mlf_logger if mlf_logger else logger_board,
        max_epochs=args['exp']['epochs'],
        enable_progress_bar=False,
        enable_model_summary=False, 
        devices=args['exp']['gpus'],
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback, alpharise]
        # callbacks=[lr_monitor]
    )
    
    # train model
    start_time = time.time()
    trainer.fit(model)
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Total training time: {training_time:.3f} seconds")
    if args.exp.logging:
        mlf_logger.log_metrics({'training_time': round(time.time() - start_time, 2)})
    best_checkpoint_path = checkpoint_callback.best_model_path
    if args['exp']['load_best']:
        model = cls.load_from_checkpoint(best_checkpoint_path, dataset_collection=dataset_collection, config=args)
        model.trainer = trainer
        model.eval()

    results = {}
    val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(dataset_collection.val_f, logger=logger)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')
    encoder_results = {}
    start_time = time.time()
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = model.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
                                                                                              one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    results.update(encoder_results)
    mlf_logger.log_metrics(encoder_results) if mlf_logger else None

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
    elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    print(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {
        'decoder_val_rmse_all': val_rmse_all,
        'decoder_val_rmse_orig': val_rmse_orig
    }
    decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})
    
    results.update(decoder_results)
    mlf_logger.log_metrics(decoder_results) if mlf_logger else None
    
    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if mlf_logger else None

    test_time = time.time() - start_time
    append_results['test_time'] = test_time
    append_results['training_time'] = training_time
    append_results['seed'] = args.exp.seed
    results.update(append_results)
    csv_dir = os.path.join(get_original_cwd(), args.exp.csv_dir)
    df = pd.DataFrame(results, index=[0])
    cols = list(df.columns)
    csv_path = os.path.join(csv_dir, f'{args.model.name}.csv')
    if not os.path.exists(csv_path):
        os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, 'w') as file:
            file.write(','.join(cols) + '\n')
    with open(csv_path, 'a') as file:
        file.write(','.join([str(results[col]) for col in cols]) + '\n')

    if args['exp']['clear_tf']:
        clear_tfevents(current_dir)

    return results
    

if __name__ == "__main__":
    main()


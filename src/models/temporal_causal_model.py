import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import logging
from typing import List
from thop import clever_format, profile
from fvcore.nn import FlopCountAnalysis
from src.models.utils_tcn import TemporalConvNet

class TemporalCausalInfModel(pl.LightningModule):
    def __init__(self, dataset_collection, config):
        super().__init__()
        self.dataset_collection = dataset_collection
        self.config = config
        self.init_params()
        self.init_model()
        self.init_ema()
        self.count_flops_processed = False
        self.automatic_optimization = False
        self.save_hyperparameters('config')
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_params(self):
        self.init_exp_params()
        self.init_dataset_params()
        self.init_model_params()

    def init_dataset_params(self):
        self.max_seq_length = self.config['dataset']['max_seq_length']
        self.treatment_size = self.config['dataset']['treatment_size']
        self.one_hot_treatment_size = self.config['dataset']['one_hot_treatment_size']
        self.static_size = self.config['dataset']['static_size']
        self.output_size = self.config['dataset']['output_size']
        self.input_size = self.config['dataset']['input_size']
        self.treatment_mode = self.config['dataset']['treatment_mode']
        self.autoregressive = self.config['dataset']['autoregressive']
        self.val_batch_size = self.config['dataset']['val_batch_size']
        self.projection_horizon = self.config['dataset']['projection_horizon']
        self.predict_X = self.config['dataset']['predict_X']
        
    def init_exp_params(self):
        self.lr = self.config['exp']['lr']
        self.lr_D = self.config['exp']['lr_D']
        self.weight_decay = self.config['exp']['weight_decay']
        self.weight_decay_D = self.config['exp']['weight_decay_D']
        self.patience = self.config['exp']['sch_patience']
        self.patience_D = self.config['exp']['sch_patience_D']
        if 'lr_X' in self.config['exp']:
            self.lr_X = self.config['exp']['lr_X']
            self.weight_decay_X = self.config['exp']['weight_decay_X']

        self.factor = self.config['exp']['factor']
        self.batch_size = self.config['exp']['batch_size']
        self.dropout = self.config['exp']['dropout']
        self.cooldown = self.config['exp']['cooldown']
        self.weights_ema = self.config['exp']['weights_ema']
        self.beta = self.config['exp']['beta']
        self.update_lambda_D = self.config['exp']['update_lambda_D']
        self.lambda_D = self.config['exp']['lambda_D'] if not self.update_lambda_D else 0.0
        self.lambda_D_max = self.config['exp']['lambda_D']
        self.lambda_X = self.config['exp']['lambda_X']
        self.lambda_Y = self.config['exp']['lambda_Y']
        self.loss_type_X = self.config['exp']['loss_type_X']
        self.epochs = self.config['exp']['epochs']

    def init_model_params(self):
        pass

    def init_model_params_(self):
        self.transpose = self.config['model']['transpose']
        if self.transpose:
            self.transpose_size = self.config['model']['transpose_size']
        self.num_blocks = self.config['model']['num_blocks']
        # check is self.num_blocks equals to 1 or 2
        if self.num_blocks not in [1, 2]:
            raise ValueError('num_blocks should be 1 or 2')

        self.first_net = self.config['model']['first_net']
        # init parameters for the first net
        if self.first_net == 'lstm':
            self.hidden_size = self.config['model']['hidden_size']
            self.num_layers = self.config['model']['num_layers']
            # self.hidden_net = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size_1, num_layers=self.num_layers_1, batch_first=True)
        elif self.first_net == 'tcn':
            self.num_channels_hidden = self.config['model']['num_channels_hidden']
            self.kernel_size = self.config['model']['kernel_size']
            # self.hidden_net = TemporalConvNet(self.input_size, self.num_channels_hidden, self.kernel_size)
        else:
            raise ValueError('first_net should be one of lstm and tcn')
        self.br_size = self.config['model']['br_size']
        self.recursive = self.config['model']['recursive']
        # init parameters for the second net if self.num_blocks == 2
        if self.num_blocks == 2:
            self.init_second_net_params_()
        # init parameters for the G_y net to predict Y
        self.hiddens_G_y = self.config['model']['hiddens_G_y']
        # init parameters for the G_x net to predict X
        if self.predict_X:
            self.hiddens_G_x = self.config['model']['hiddens_G_x']
        self.ema_y = self.config['model']['ema_y']
        self.init = self.config['model']['init']

    def init_second_net_params_(self):
        self.second_net = self.config['model']['second_net']
        if self.second_net == 'lstm':
            if self.predict_X:
                self.hidden_size_x = self.config['model']['hidden_size_x']
                self.num_layers_x = self.config['model']['num_layers_x']
            self.hidden_size_y = self.config['model']['hidden_size_y']
            self.num_layers_y = self.config['model']['num_layers_y']
        elif self.second_net == 'tcn':
            if self.predict_X:
                self.num_channels_hidden_x = self.config['model']['num_channels_hidden_x']
                self.kernel_size_x = self.config['model']['kernel_size_x']
            self.num_channels_hidden_y = self.config['model']['num_channels_hidden_y']
            self.kernel_size_y = self.config['model']['kernel_size_y']
        else:
            raise ValueError('second_net should be one of lstm and tcn')

    def init_model(self):
        pass

    def init_model_(self):
        if self.static_size > 0:
            input_size = self.input_size + self.static_size + self.treatment_size
        if self.autoregressive:
            # we need to use the previous output as the input
            input_size += self.output_size
        # init the transpose net to transpose the input if needed
        if self.transpose:
            self.transpose_net = nn.Sequential()
            self.transpose_net.add_module('linear1', nn.Linear(input_size, self.transpose_size))
            # self.transpose_net.add_module('elu1', nn.ELU())
            input_size = self.transpose_size
        else:
            self.transpose_net = nn.Identity()

        # init the first net
        if self.first_net == 'lstm':
            self.hidden_net = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            input_size = self.hidden_size
        elif self.first_net == 'tcn':
            self.hidden_net = TemporalConvNet(input_size, self.num_channels_hidden, self.kernel_size, dropout=self.dropout, init=self.init)
            input_size = self.num_channels_hidden[-1]
        else:
            raise ValueError('first_net should be one of lstm and tcn')
        # init the G_br net to learn the balancing representation
        self.G_br = nn.Sequential()
        self.G_br.add_module('linear1', nn.Linear(input_size, self.br_size))
        if self.config['model']['br_act']:
            self.G_br.add_module('elu1', nn.ELU())
        # init the second net if self.num_blocks == 2
        if self.num_blocks == 2:
            input_size_x, input_size_y = self.init_second_net_()
        else:
            input_size_x, input_size_y = self.br_size + self.treatment_size, self.br_size + self.treatment_size

        # init the G_y net to predict Y
        self.G_y = nn.Sequential()
        for i in range(len(self.hiddens_G_y)):
            if i == 0:
                self.G_y.add_module('fc{}'.format(i), nn.Linear(input_size_y, self.hiddens_G_y[i]))
            else:
                self.G_y.add_module('elu{}'.format(i), nn.ELU())
                self.G_y.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_y[i-1], self.hiddens_G_y[i]))
        self.G_y.add_module('elu{}'.format(len(self.hiddens_G_y)), nn.ELU())
        self.G_y.add_module('fc{}'.format(len(self.hiddens_G_y)), nn.Linear(self.hiddens_G_y[-1], self.output_size))
        # init the G_x net to predict X if needed
        if self.predict_X:
            self.G_x = nn.Sequential()
            for i in range(len(self.hiddens_G_x)):
                if i == 0:
                    self.G_x.add_module('fc{}'.format(i), nn.Linear(input_size_x, self.hiddens_G_x[i]))
                else:
                    self.G_x.add_module('elu{}'.format(i), nn.ELU())
                    self.G_x.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_x[i-1], self.hiddens_G_x[i]))
            self.G_x.add_module('elu{}'.format(len(self.hiddens_G_x)), nn.ELU())
            self.G_x.add_module('fc{}'.format(len(self.hiddens_G_x)), nn.Linear(self.hiddens_G_x[-1], self.input_size))
        else:
            self.G_x = nn.Identity()
        # init the ema_net_x to predict X if needed
        # the ema_net_x will be used to cauculate the ema of the predicted X
        if self.predict_X:
            self.ema_net_x = nn.Sequential()
            if 'hiddens_ema' in self.config['model']:
                hiddens_ema = self.config['model']['hiddens_ema']
                for i in range(len(hiddens_ema)):
                    if i == 0:
                        self.ema_net_x.add_module('fc{}'.format(i), nn.Linear(input_size_x, hiddens_ema[i]))
                    else:
                        self.ema_net_x.add_module('elu{}'.format(i), nn.ELU())
                        self.ema_net_x.add_module('fc{}'.format(i), nn.Linear(hiddens_ema[i-1], hiddens_ema[i]))
                self.ema_net_x.add_module('elu{}'.format(len(hiddens_ema)), nn.ELU())
                self.ema_net_x.add_module('fc{}'.format(len(hiddens_ema)), nn.Linear(hiddens_ema[-1], self.input_size))
                self.ema_net_x.add_module('sigmoid{}'.format(1), nn.Sigmoid())
            else:
                self.ema_net_x.add_module('fc{}'.format(1), nn.Linear(input_size_x, self.input_size))
                self.ema_net_x.add_module('sigmoid{}'.format(1), nn.Sigmoid())
        else:
            self.ema_net_x = nn.Identity()

        if self.config['model']['ema_y']:
            self.ema_net_y = nn.Sequential()
            self.ema_net_y.add_module('fc{}'.format(1), nn.Linear(input_size_y, self.output_size))
            self.ema_net_y.add_module('sigmoid{}'.format(1), nn.Sigmoid())
        else:
            self.ema_net_y = nn.Identity()

    def init_second_net_(self):
        # check self.recursive, if True, we will use the current A as the input of the second net, otherwise we will use the current A as the input of G_x and G_y
        if self.recursive:
            input_size = self.br_size + self.treatment_size 
        else:
            input_size = self.br_size
        input_size_x = 0
        if self.second_net == 'lstm':
            if self.predict_X:
                self.hidden_net_x = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size_x, num_layers=self.num_layers_x, batch_first=True)
                input_size_x = self.hidden_size_x
            else:
                self.hidden_net_x = nn.Identity()
            self.hidden_net_y = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size_y, num_layers=self.num_layers_y, batch_first=True)
            input_size_y = self.hidden_size_y
        elif self.second_net == 'tcn':
            if self.predict_X:
                if 'dropout_x' in self.config['exp']:
                    dropout = self.config['exp']['dropout_x']
                else:
                    dropout = self.dropout
                self.hidden_net_x = TemporalConvNet(input_size, self.num_channels_hidden_x, self.kernel_size_x, dropout=dropout, init=self.init)
                input_size_x = self.num_channels_hidden_x[-1]
            else:
                self.hidden_net_x = nn.Identity()
            self.hidden_net_y = TemporalConvNet(input_size, self.num_channels_hidden_y, self.kernel_size_y, dropout=self.dropout, init=self.init)
            input_size_y = self.num_channels_hidden_y[-1]
        else:
            raise ValueError('second_net should be one of lstm and tcn')

        if not self.recursive:
            if self.predict_X:
                input_size_x += self.treatment_size 
            input_size_y += self.treatment_size 
        return input_size_x, input_size_y

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.val_f, batch_size=self.val_batch_size)

    def count_flops(self, batch):
        flops = FlopCountAnalysis(self, batch)
        # print('FLOPs:', flops.total())
        mflops = flops.total() / 1e6 / self.batch_size
        print('FLOPs: {:.2f} MFLOPs'.format(mflops))
        for name, module_flops in flops.by_module().items():
            module_flops_per_sample = module_flops / 1e6 / self.batch_size
            # print(f'{name} : {module_flops_per_sample:.2f} MFLOPs')
        params = sum(p.numel() for p in self.parameters())
        print('Parameters:', params)
        return params, mflops

    def forward(self, x):
        # you should implement the forward pass specifically for your model.
        # this returns the concatenation of y_hat and x_hat
        pass

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward, but iis dependent on the strategy chosen for balancing.
        pass

    def validation_step(self, batch, batch_idx):
        # validation_step defined the train loop.
        # It is independent of forward, but iis dependent on the strategy chosen for balancing.
        pass

    def get_mse_at_follow_up_time(self, prediction, output, active_entries=None):
        # cauculate mse at follow up time
        mses = torch.sum(torch.sum((prediction - output) ** 2 * active_entries, dim=0), dim=-1) / torch.sum(torch.sum(active_entries, dim=0), dim=-1)
        return mses

    def get_mse_all(self, prediction, output, active_entries=None):
        mses = torch.sum((prediction - output) ** 2 * active_entries) / torch.sum(active_entries)
        return mses

    def get_l1_all(self, prediction, output, active_entries=None):
        l1 = torch.sum(torch.abs(prediction - output) * active_entries) / torch.sum(active_entries)
        return l1

    def get_predictions(self, dataset: Dataset, logger=None) -> np.array:
        if logger is not None:
            logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams['config']['dataset']['val_batch_size'], shuffle=False)
        outcome_pred, next_covariates_pred = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy(), next_covariates_pred.numpy()

    def get_normalised_masked_rmse(self, dataset: Dataset, one_step_counterfactual=False, logger=None):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'RMSE calculation for {dataset.subset_name}.')
        outputs_scaled, _ = self.get_predictions(dataset, logger=logger)
        
        unscale = self.hparams['config']['exp']['unscale_rmse']
        percentage = self.hparams['config']['exp']['percentage_rmse']

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * dataset.data['active_entries']
        else:
            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_scaled - dataset.data['outputs']) ** 2) * dataset.data['active_entries']

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
        mse_orig = mse_orig.mean()
        rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        # Masked averaging over all dimensions at once
        mse_all = mse.sum() / dataset.data['active_entries'].sum()
        rmse_normalised_all = np.sqrt(mse_all) / dataset.norm_const

        if percentage:
            rmse_normalised_orig *= 100.0
            rmse_normalised_all *= 100.0

        if one_step_counterfactual:
            # Only considering last active entry with actual counterfactuals
            num_samples, time_dim, output_dim = dataset.data['active_entries'].shape
            last_entries = dataset.data['active_entries'] - np.concatenate([dataset.data['active_entries'][:, 1:, :], np.zeros((num_samples, 1, output_dim))], axis=1)
            if unscale:
                mse_last = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * last_entries
            else:
                mse_last = ((outputs_scaled - dataset.data['outputs']) ** 2) * last_entries

            mse_last = mse_last.sum() / last_entries.sum()
            rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

            if percentage:
                rmse_normalised_last *= 100.0

            return rmse_normalised_orig, rmse_normalised_all, rmse_normalised_last

        return rmse_normalised_orig, rmse_normalised_all

    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc: List[Dataset] = None, logger=None):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'RMSE calculation for {dataset.subset_name}.')
        assert hasattr(dataset, 'data_processed_seq')

        unscale = self.hparams['config']['exp']['unscale_rmse']
        percentage = self.hparams['config']['exp']['percentage_rmse']
        outputs_scaled = self.get_autoregressive_predictions(dataset if datasets_mc is None else datasets_mc, logger=logger)

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs']) ** 2) \
                * dataset.data_processed_seq['active_entries']
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs']) ** 2) * dataset.data_processed_seq['active_entries']

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse[not_nan].sum(0).sum(-1) / dataset.data_processed_seq['active_entries'][not_nan].sum(0).sum(-1)
        rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        if percentage:
            rmses_normalised_orig *= 100.0

        return rmses_normalised_orig

    def get_autoregressive_predictions(self, dataset: Dataset, logger=None) -> np.array:
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        projection_horizon = self.hparams['config']['dataset']['projection_horizon']

        predicted_outputs = np.zeros((len(dataset), projection_horizon, self.output_size))

        for t in range(projection_horizon + 1):
            if logger is not None:
                logger.info(f't = {t + 1}')
            outputs_scaled, next_covariates_pred = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data['future_past_split'][i])
                if t < projection_horizon:
                    if self.predict_X:
                        # replace the covariates in next step with the predicted covariates
                        dataset.data['vitals'][i, split + t, :] = next_covariates_pred[i, split - 1 + t, :]
                    dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                    pass

                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs
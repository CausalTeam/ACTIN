import numpy as np
from torch.utils.data import Dataset
import logging
from copy import deepcopy
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset_collection import SyntheticDatasetCollection
from src.data.CI_sim.sim import simulate_factual, simulate_counterfactual_one_step, simulate_counterfactuals_treatment_seq

logger = logging.getLogger(__name__)

class SyntheticCIDataset(Dataset):
    # dataset for continuous intervention
    def __init__(self,
        noise_scale_y: float,
        noise_scale_x: float,
        noise_scale_a: float,
        theta_x_sum: float,
        size: int,
        gamma: float,
        mode_x: str,
        mode_a: str,
        seq_length: int,
        projection_horizon: int,
        lag: int,
        subset_name: str,
        norm_const: float = 1.0,
        mode='factual',
        cf_seq_mode='sliding_treatment',
        treatment_mode='multiclass',
        ):
        self.noise_scale_y = noise_scale_y
        self.noise_scale_x = noise_scale_x
        self.noise_scale_a = noise_scale_a
        self.theta_x_sum = theta_x_sum
        self.size = size
        self.gamma = gamma
        self.mode_x = mode
        self.mode_a = mode_a
        self.seq_length = seq_length
        self.projection_horizon = projection_horizon
        self.lag = lag
        self.subset_name = subset_name
        self.cf_seq_mode = cf_seq_mode
        self.treatment_mode = treatment_mode
        self.mode = mode
        self.norm_const = norm_const
        self.params = {
            'theta_x': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            'theta_y': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            'noise_scale_y': noise_scale_y,
            'noise_scale_x': noise_scale_x,
            'noise_scale_a': noise_scale_a,
            'gamma': gamma
        }
        self.params['theta_x'] *= (1 / np.sum(self.params['theta_x'])) * theta_x_sum

        if mode == 'factual':
            self.data = simulate_factual(size, seq_length, lag, self.params, mode_x=mode_x, mode_a=mode_a)
            print('successfully generate factual data of {}'.format(subset_name))
        elif mode == 'counterfactual':
            # with this mode, we can generate intervention randomly, i.e., mode_a='v1'
            self.data = simulate_factual(size, seq_length, lag, self.params, mode_x=mode_x, mode_a='v1')
            print('successfully generate counterfactual data of {}'.format(subset_name))
        elif mode == 'counterfactual_one_step':
            self.data = simulate_counterfactual_one_step(size, seq_length, lag, self.params, mode_x=mode_x, mode_a=mode_a)
            print('successfully generate counterfactual one step data of {}'.format(subset_name))
        elif mode == 'counterfactual_treatment_seq':
            assert projection_horizon is not None
            self.data = simulate_counterfactuals_treatment_seq(size, seq_length, lag, self.params, projection_horizon=projection_horizon, mode_x=mode_x, mode_a=mode_a)
        self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.treatment_mode = treatment_mode
        self.exploded = False

    def __len__(self):
        return self.data['current_covariates'].shape[0]

    def __getitem__(self, index) -> dict:
        # result = {k: v[index] for k, v in self.data.items() if hasattr(v, '__len__') and len(v) == len(self)}
        result = {k: v[index] for k, v in self.data.items()}
        
        return result

    def process_data(self):
        # prepare data for one-step-head prediction
        # used by CT
        if self.processed:
            logger.info(f'{self.subset_name} Dataset already processed')
        else:
            logger.info(f'Processing {self.subset_name} Dataset')
            self.data['current_covariates'] = self.data['current_covariates'].astype(np.float32)
            self.data['next_covariates'] = self.data['next_covariates'].astype(np.float32)
            self.data['prev_treatments'] = self.data['prev_treatments'].astype(np.float32)
            self.data['current_treatments'] = self.data['current_treatments'].astype(np.float32)
            self.data['outputs'] = self.data['outputs'].astype(np.float32)
            self.data['vitals'] = self.data['current_covariates']
            self.data['next_vitals'] = self.data['next_covariates'][:, :-1, :]
            self.data['prev_outputs'] = np.concatenate((np.zeros((self.data['outputs'].shape[0], 1, 1)), self.data['outputs'][:, :-1, :]), axis=1)
            self.processed = True

    def process_sequential_test(self, projection_horizon, encoder_r=None, save_encoder_r=False):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        """
        Pre-process test dataset for multiple-step-ahead prediction: takes the last size-steps according to the projection horizon
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before testing (multiple sequences)')
            sequence_lengths = self.data['sequence_lengths']
            outputs = self.data['outputs']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments'][:, 1:, :]  # Without zero_init_treatment
            current_covariates = self.data['current_covariates']

            size, max_seq_length, num_features = outputs.shape

            if encoder_r is not None:
                seq2seq_state_inits = np.zeros((size, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((size, max_seq_length - projection_horizon))
            seq2seq_previous_treatments = np.zeros((size, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((size, projection_horizon, current_treatments.shape[-1]))
            seq2seq_current_covariates = np.zeros((size, projection_horizon, current_covariates.shape[-1]))
            seq2seq_outputs = np.zeros((size, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((size, projection_horizon, 1))
            seq2seq_sequence_lengths = np.zeros(size)

            seq2seq_outputs_prev = np.zeros((size, projection_horizon, outputs.shape[-1]))

            for i in range(size):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                if encoder_r is not None:
                    seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
                seq2seq_active_encoder_r[i, :fact_length] = 1.0

                seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
                seq2seq_previous_treatments[i] = previous_treatments[i, fact_length - 1:fact_length + projection_horizon - 1, :]
                seq2seq_current_treatments[i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]
                seq2seq_outputs[i] = outputs[i, fact_length: fact_length + projection_horizon, :]
                seq2seq_sequence_lengths[i] = projection_horizon
                # Disabled teacher forcing for test dataset
                seq2seq_current_covariates[i] = np.repeat([current_covariates[i, fact_length - 1]], projection_horizon, axis=0)

                seq2seq_outputs_prev[i] = outputs[i, fact_length - 1: fact_length + projection_horizon - 1, :]

            # Package outputs
            seq2seq_data = {
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'current_covariates': seq2seq_current_covariates,
                'prev_outputs': seq2seq_outputs_prev,
                'static_features': self.data['static_features'],
                'outputs': seq2seq_outputs,
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
            }
            if encoder_r is not None:
                seq2seq_data['init_state'] = seq2seq_state_inits

            self.data_original = deepcopy(self.data)
            self.data = seq2seq_data
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r and encoder_r is not None:
                self.encoder_r = encoder_r[:, :max_seq_length - projection_horizon, :]

            self.processed_sequential = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_sequential_multi(self, projection_horizon):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        """
        Pre-process test dataset for multiple-step-ahead prediction for multi-input model: marking rolling origin with
            'future_past_split'
        Args:
            projection_horizon: Projection horizon
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            self.data_processed_seq = self.data
            self.data = deepcopy(self.data_original)
            self.data['future_past_split'] = self.data['sequence_lengths'] - projection_horizon
            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data

    def process_sequential_vcnet(self, projection_horizon):
        assert self.processed_sequential

        if not self.processed_autoregressive:
            self.data_processed_seq = self.data
            self.data = deepcopy(self.data_original)
            self.data['future_past_split'] = self.data['sequence_lengths'] - projection_horizon
            self.processed_autoregressive = True
            # set the covariates after point future_past_split to 0
            for i in range(self.data['current_covariates'].shape[0]):
                future_past_split = int(self.data['future_past_split'][i])
                self.data['current_covariates'][i, future_past_split:, :] = 0
                self.data['next_covariates'][i, future_past_split:, :] = 0

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data


class SyntheticCIDatasetCollection(SyntheticDatasetCollection):
    def __init__(self,
        noise_scale_y: float,
        noise_scale_x: float,
        noise_scale_a: float,
        theta_x_sum: float,
        gamma: float,
        data_size: dict,
        mode_x: str,
        mode_a: str,
        max_seq_length=23,
        projection_horizon=5,
        lag=3,
        cf_seq_mode='sliding_treatment',
        treatment_mode='multiclass',
        **kwargs
        ):
        # super(SyntheticCIDatasetCollection, self).__init__()
        super().__init__(**kwargs)
        # self.seed = seed
        # np.random.seed(10)
        print('building dataset collection')
        self.train_f = SyntheticCIDataset(noise_scale_y, noise_scale_x, noise_scale_a, theta_x_sum, data_size['train'], gamma, mode_x, mode_a, max_seq_length, projection_horizon, lag, 'train_f', mode='factual', cf_seq_mode=cf_seq_mode, treatment_mode=treatment_mode)
        self.val_f = SyntheticCIDataset(noise_scale_y, noise_scale_x, noise_scale_a, theta_x_sum, data_size['val'], gamma, mode_x, mode_a, max_seq_length, projection_horizon, lag, 'val_f', mode='factual', cf_seq_mode=cf_seq_mode, treatment_mode=treatment_mode)
        self.test_cf = SyntheticCIDataset(noise_scale_y, noise_scale_x, noise_scale_a, theta_x_sum, data_size['test'], gamma, mode_x, mode_a, max_seq_length, projection_horizon, lag, 'test_cf', mode='counterfactual', cf_seq_mode=cf_seq_mode, treatment_mode=treatment_mode)
        self.test_cf_one_step = SyntheticCIDataset(noise_scale_y, noise_scale_x, noise_scale_a, theta_x_sum, data_size['test'], gamma, mode_x, mode_a, max_seq_length, projection_horizon, lag, 'test_cf_one_step', mode='counterfactual_one_step', cf_seq_mode=cf_seq_mode, treatment_mode=treatment_mode)
        self.test_cf_treatment_seq = SyntheticCIDataset(noise_scale_y, noise_scale_x, noise_scale_a, theta_x_sum, data_size['test'], gamma, mode_x, mode_a, max_seq_length, projection_horizon, lag, 'test_cf_treatment_seq', mode='counterfactual_treatment_seq', cf_seq_mode=cf_seq_mode, treatment_mode=treatment_mode)
        
        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = False

    def process_data_multi(self):
        """
        Used by CT
        """
        self.train_f.process_data()
        if hasattr(self, 'val_f') and self.val_f is not None:
            self.val_f.process_data()
        if hasattr(self, 'test_cf') and self.test_cf is not None:
            self.test_cf.process_data()
        if hasattr(self, 'test_cf_one_step') and self.test_cf_one_step is not None:
            self.test_cf_one_step.process_data()
        if hasattr(self, 'test_cf_treatment_seq') and self.test_cf_treatment_seq is not None:
            self.test_cf_treatment_seq.process_data()
            self.test_cf_treatment_seq.process_sequential_test(self.projection_horizon)
            self.test_cf_treatment_seq.process_sequential_vcnet(self.projection_horizon)

        # self.test_cf_one_step.process_data(self.train_scaling_params)
        # self.test_cf_treatment_seq.process_data(self.train_scaling_params)
        # self.test_cf_treatment_seq.process_sequential_test(self.projection_horizon)
        # self.test_cf_treatment_seq.process_sequential_multi(self.projection_horizon)

        self.processed_data_multi = True

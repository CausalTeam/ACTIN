import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.balancing_representation_model import CausalBrModel
from torch_ema import ExponentialMovingAverage
from torch.nn.utils import clip_grad_norm_
from src.utils.utils import sample_treatments, advanced_indexing_shuffle_3d
import numpy as np

class CausalGanBrModel(CausalBrModel):
    def __init__(self, dataset_collection, config):
        super().__init__(dataset_collection, config)

    def init_ema(self):
        if self.weights_ema:
            other_parameters, balancing_params = self.split_parameters()
            self.ema_treatment = ExponentialMovingAverage([par for par in balancing_params], decay=self.beta)
            self.ema_non_treatment = ExponentialMovingAverage([par for par in other_parameters], decay=self.beta)

    def init_model_params(self):
        self.init_model_params_()
        # init parameters for the D_net net to predict A
        self.hiddens_D_net = self.config['model']['hiddens_D_net']
        self.balancing = self.config['model']['balancing']

    def init_model(self):
        self.init_model_()
        # init the D_net to predict A
        self.D_net = nn.Sequential()
        if self.balancing == 'confuse':
            input_size = self.br_size 
            output_size = self.treatment_size
        elif 'mine' in self.balancing:
            input_size = self.br_size + self.treatment_size
            output_size = 1
        for i in range(len(self.hiddens_D_net)):
            if i == 0:
                self.D_net.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_D_net[i]))
            else:
                self.D_net.add_module('elu{}'.format(i), nn.ELU())
                self.D_net.add_module('fc{}'.format(i), nn.Linear(self.hiddens_D_net[i-1], self.hiddens_D_net[i]))
        self.D_net.add_module('elu{}'.format(len(self.hiddens_D_net)), nn.ELU())
        self.D_net.add_module('fc{}'.format(len(self.hiddens_D_net)), nn.Linear(self.hiddens_D_net[-1], output_size))

    def get_a_hat(self, batch, update_D=True):
        # get the predicted A
        # if update_D is True, we will only update the D_net
        br = self.build_br(batch)
        if update_D:
            # we will not update the hidden_net if update_D
            br = br.detach()
        a_hat = self.D_net(br)
        return a_hat

    def cauculate_mi_loss(self, batch, update_D=True):
        # if update_D is True, we will only update the D_net
        br = self.build_br(batch)
        if update_D:
            # we will not update the hidden_net if update_D
            br = br.detach()
        
        current_treatments = batch['current_treatments']
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(current_treatments)
        factual_samples = torch.cat([br, current_treatments], dim=-1)
        if self.balancing == 'mine-random':
            random_treatments = sample_treatments(current_treatments, self.treatment_mode).to(self.device)
        else:
            random_treatments = advanced_indexing_shuffle_3d(current_treatments, self.config['model']['shffule_mode'])
        counterfactual_samples = torch.cat([br, random_treatments], dim=-1)
        labels = torch.cat([torch.ones_like(active_entries), torch.zeros_like(active_entries)], dim=0)
        samples = torch.cat([factual_samples, counterfactual_samples], dim=0)
        logits = self.D_net(samples)
        if self.balancing == 'mine-confuse':
            if update_D:
                loss = F.mse_loss(F.sigmoid(logits), labels, reduce=False)
            else:
                uniform_labels = torch.ones_like(labels) * 0.5
                loss = F.mse_loss(F.sigmoid(logits), uniform_labels, reduce=False)
        else:
            loss = F.mse_loss(F.sigmoid(logits), labels, reduce=False)
        # loss = F.mse_loss(logits, labels, reduce=False)
        active_entries = torch.cat([active_entries, active_entries], dim=0)
        loss = torch.sum(loss * active_entries) / torch.sum(active_entries)

        return loss

    def bce(self, treatment_pred, current_treatments, active_entries):
        if self.treatment_mode == 'multiclass':
            loss = F.cross_entropy(treatment_pred.permute(0, 2, 1), current_treatments.permute(0, 2, 1), reduce=False)
            loss = loss.unsqueeze(-1)
        elif self.treatment_mode == 'multilabel':
            loss = F.binary_cross_entropy_with_logits(treatment_pred, current_treatments, reduce=False)
        else:
            raise NotImplementedError()
        loss = torch.sum(loss * active_entries) / torch.sum(active_entries)
        return loss

    def bce_loss(self, treatment_pred, current_treatments, active_entries, kind='predict'):
        if kind == 'predict':
            bce_loss = self.bce(treatment_pred, current_treatments, active_entries)
        elif kind == 'confuse':
            uniform_treatments = torch.ones_like(current_treatments)
            if self.treatment_mode == 'multiclass':
                uniform_treatments *= 1 / current_treatments.shape[-1]
            elif self.treatment_mode == 'multilabel':
                uniform_treatments *= 0.5
            bce_loss = self.bce(treatment_pred, uniform_treatments, active_entries)
        else:
            raise NotImplementedError()
        return bce_loss

    def training_step(self, batch, batch_idx):
        # Calculate FLOPs only for the first batch and the first epoch
        if batch_idx == 0 and not self.count_flops_processed:
            # Calculate FLOPs using FlopCountAnalysis
            params, mflops = self.count_flops(batch)
            self.log('params', params)
            self.log('mflops', mflops)
            self.count_flops_processed = True

        optimizer_D, optimizer_O = self.optimizers()
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)
        if self.lambda_D > 0:
            # update the discriminator D
            self.toggle_optimizer(optimizer_D)
            for param in self.D_net.parameters():
                param.requires_grad = True

            if self.balancing == 'confuse':
                a_hat = self.get_a_hat(batch)
                current_treatments = batch['current_treatments']
                # loss_D = self.cauculate_loss_D(a_hat, current_treatments, active_entries)
                loss_D = self.bce_loss(a_hat, current_treatments, active_entries)
            elif 'mine' in self.balancing:
                loss_D = self.cauculate_mi_loss(batch)

            self.manual_backward(loss_D)
            clip_grad_norm_(self.D_net.parameters(), max_norm=1.0)
            optimizer_D.step()
            optimizer_D.zero_grad()
            if self.weights_ema:
                self.ema_treatment.update()
            self.untoggle_optimizer(optimizer_D)

        # update the other parameters
        self.toggle_optimizer(optimizer_O)
        for param in self.D_net.parameters():
            param.requires_grad = False

        br = self.build_br(batch)
        y_hat = self.forward_y(batch, br)
        output = batch['outputs']
        loss_y = self.get_mse_all(y_hat, output, active_entries)
        # if self.trainer.current_epoch % 30 == 29:
        if self.lambda_D > 0:
            if self.balancing == 'confuse':
                a_hat = self.get_a_hat(batch, False)
                loss_D = self.bce_loss(a_hat, current_treatments, active_entries, kind='confuse')
            elif 'mine' in self.balancing:
                if self.balancing == 'mine-confuse':
                    loss_D = self.cauculate_mi_loss(batch, False)
                else:
                    loss_D = - self.cauculate_mi_loss(batch, False)
        else:
            loss_D = 0

        if self.predict_X:
            br = self.build_br(batch)
            next_covariates = batch['next_vitals']
            x_hat = self.forward_x(batch, br)[:, :next_covariates.shape[1]]
            if self.loss_type_X == 'l1':
                loss_x = self.get_l1_all(x_hat, next_covariates, active_entries[:, 1:])
            elif self.loss_type_X == 'l2':
                loss_x = self.get_mse_all(x_hat, next_covariates, active_entries[:, 1:])
            else:
                raise ValueError('loss_type_X should be one of l1 and l2')
        else:
            loss_x = 0
        loss = self.lambda_Y * loss_y + self.lambda_X * loss_x + self.lambda_D * loss_D

        self.manual_backward(loss)
        clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer_O.step()
        optimizer_O.zero_grad()
        self.untoggle_optimizer(optimizer_O)

        if self.weights_ema:
            self.ema_non_treatment.update()
            
        self.log('train_loss', loss)
        self.log('loss_x', loss_x, on_epoch=True)
        self.log('loss_y', loss_y, on_epoch=True)
        return {'loss': loss, 'loss_x': loss_x, 'loss_D': loss_D}

    def validation_step(self, batch, batch_idx):
        if self.weights_ema:
            with self.ema_non_treatment.average_parameters():
                with self.ema_treatment.average_parameters():
                    out = self.forward(batch)
        else:
            out = self.forward(batch)
        output = batch['outputs']
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)
        y_hat = out[:,:,:self.output_size]
        loss_y = self.get_mse_all(y_hat, output, active_entries)

        if self.predict_X:
            next_covariates = batch['next_vitals']
            x_hat = out[:, :next_covariates.shape[1], self.output_size:]
            if self.loss_type_X == 'l1':
                loss_x = self.get_l1_all(x_hat, next_covariates, active_entries[:, 1:])
            elif self.loss_type_X == 'l2':
                loss_x = self.get_mse_all(x_hat, next_covariates, active_entries[:, 1:])
        else:
            loss_x = 0
        
        # loss = loss_y + loss_x * 0.1
        loss = self.lambda_Y * loss_y + self.lambda_X * loss_x

        if 'mine' in self.balancing:
            loss_D = - self.cauculate_mi_loss(batch, True)
            # self.print_avg_gradients(self.D_net) 
        elif self.balancing == 'confuse':
            a_hat = self.get_a_hat(batch, False)
            current_treatments = batch['current_treatments']
            loss_D = self.bce_loss(a_hat, current_treatments, active_entries, kind='confuse')
        
        self.log('val_loss_D', loss_D, on_epoch=True)

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_loss_x', loss_x, on_epoch=True)
        self.log('val_loss_y', loss_y, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        if self.weights_ema:
            with self.ema_non_treatment.average_parameters():
                with self.ema_treatment.average_parameters():
                    out = self.forward(batch)
        else:
            out = self.forward(batch)
        
        prediction = out[:,:,:self.output_size]
        next_covariates = out[:,:,self.output_size:]
        return prediction, next_covariates
    
    def split_parameters(self):
        # split the parameters into three parts
        other_parameters = []
        other_parameters.extend(self.transpose_net.parameters())
        other_parameters.extend(self.hidden_net.parameters())
        other_parameters.extend(self.G_br.parameters())
        if self.num_blocks == 2:
            other_parameters.extend(self.hidden_net_y.parameters())
            other_parameters.extend(self.hidden_net_x.parameters())
        other_parameters.extend(self.G_y.parameters())
        other_parameters.extend(self.ema_net_y.parameters())
        other_parameters.extend(self.G_x.parameters())
        other_parameters.extend(self.ema_net_x.parameters())
        balancing_params = list(self.D_net.parameters())
        return other_parameters, balancing_params

    def on_train_epoch_end(self):
        if 'lr_X' in self.config['exp']:
            lr_scheduler_D, lr_scheduler_O, lr_scheduler_X = self.lr_schedulers()
            lr_scheduler_X.step(self.trainer.callback_metrics["val_loss_x"])
        else:
            lr_scheduler_D, lr_scheduler_O = self.lr_schedulers()
        lr_scheduler_D.step(self.trainer.callback_metrics["val_loss"])
        lr_scheduler_O.step(self.trainer.callback_metrics["val_loss"])
        if 'loss_x_epoch' in self.trainer.logged_metrics:
            print(f"Epoch {self.trainer.current_epoch}: Valid Loss_x = {self.trainer.logged_metrics['val_loss_x']:.4f}")
            print(f"Epoch {self.trainer.current_epoch}: Valid Loss_y = {self.trainer.logged_metrics['val_loss_y']:.4f}")
            print(f"Epoch {self.trainer.current_epoch}: Valid Loss_D = {self.trainer.logged_metrics['val_loss_D']:.4f}")

    def configure_optimizers(self):
        other_parameters, balancing_params = self.split_parameters()
        optimizer_D = torch.optim.Adam(balancing_params, lr=self.lr_D, weight_decay=self.weight_decay_D)
        optimizer_O = torch.optim.Adam(other_parameters, lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler_D = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=self.factor, patience=self.patience_D, verbose=True, cooldown=self.cooldown),
            'monitor': 'val_loss',  
        }
        lr_scheduler_O = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_O, mode='min', factor=self.factor, patience=self.patience, verbose=True, cooldown=self.cooldown),
            'monitor': 'val_loss',
        }

        if self.weights_ema:
            self.ema_treatment = ExponentialMovingAverage([par for par in balancing_params], decay=self.beta)
            self.ema_non_treatment = ExponentialMovingAverage([par for par in other_parameters], decay=self.beta)
    
        return [optimizer_D, optimizer_O], [lr_scheduler_D, lr_scheduler_O]

    def on_save_checkpoint(self, checkpoint):
        # save the ema state
        if self.weights_ema:
            checkpoint['ema_treatment_state'] = self.ema_treatment.state_dict()
            checkpoint['ema_non_treatment_state'] = self.ema_non_treatment.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # load the ema state
        if self.weights_ema:
            self.ema_treatment.load_state_dict(checkpoint['ema_treatment_state'])
            self.ema_non_treatment.load_state_dict(checkpoint['ema_non_treatment_state'])

    def print_avg_gradients(self, net):
        total_grad = 0
        num_params = 1
        for param in net.parameters():
            if param.grad is not None:
                total_grad += param.grad.abs().mean()  
                num_params += 1
        avg_grad = total_grad / num_params
        print(f"Average gradient of D_net: {avg_grad}")

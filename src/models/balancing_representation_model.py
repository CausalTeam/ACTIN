from src.models.temporal_causal_model import TemporalCausalInfModel
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import torch

class CausalBrModel(TemporalCausalInfModel):
    # learning balancing representation using GAN
    def __init__(self, dataset_collection, config):
        super().__init__(dataset_collection, config)

    def build_br(self, batch):
        # build the balancing representation
        # the input of this function is the original batch
        # the output of this function is the balancing representation
        if self.static_size > 0:
            if self.predict_X:
                x = batch['vitals']
                x = torch.cat((x, batch['static_features']), dim=-1)
            # when we don't predict x, we use static features as the current_covariates
            else:
                x = batch['static_features']
        # if we use autoregressive, we need to use the previous output as the input
        if self.autoregressive:
            prev_outputs = batch['prev_outputs']
            x = torch.cat((x, prev_outputs), dim=-1)
        
        previous_treatments = batch['prev_treatments']
        x = torch.cat((x, previous_treatments), dim=-1)
        # transpose the input if needed
        if self.transpose:
            x = self.transpose_net(x)
        if self.first_net == 'lstm':
            br, _ = self.hidden_net(x)
        elif self.first_net == 'tcn':
            br = self.hidden_net(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError('first_net should be one of lstm and tcn')
        # transform the br using the G_br net
        br = self.G_br(br)
        return br

    def forward(self, batch):
        # get the balancing representation
        br = self.build_br(batch)
        # get the predicted Y
        y_hat = self.forward_y(batch, br)
        # get the predicted X if needed
        n, T, _ = br.shape
        x_hat = torch.zeros(n, T, self.input_size).to(self.device) 
        if self.predict_X:
            x_hat = self.forward_x(batch, br)
        return torch.cat((y_hat, x_hat), dim=-1)

    def forward_x(self, batch, br):
        if self.num_blocks == 2:
            out_x = self.forward_second_blocks_x(batch, br)
        else:
            out_x = torch.cat((br, batch['current_treatments']), dim=-1)
        # x_hat = self.G_x(out_x)
        out_x_reshaped = out_x.reshape(-1, out_x.shape[-1])
        x_hat_reshaped = self.G_x(out_x_reshaped)
        b, T, _ = out_x.shape
        x_hat = x_hat_reshaped.reshape(b, T, -1)
        ema_xw = self.ema_net_x(out_x)
        x_hat = ema_xw * batch['vitals'] + (1 - ema_xw) * x_hat
        # x_hat = ema_xw * batch['current_covariates'] + x_hat
        return x_hat

    def forward_y(self, batch, br):
        if self.num_blocks == 2:
            out_y = self.forward_second_blocks_y(batch, br)
        else:
            out_y = torch.cat((br, batch['current_treatments']), dim=-1)
        # y_hat = self.G_y(out_y)
        out_y_reshaped = out_y.reshape(-1, out_y.shape[-1])  
        y_hat_reshaped = self.G_y(out_y_reshaped)  
        b, T, _ = out_y.shape
        y_hat = y_hat_reshaped.reshape(b, T, -1)  
        if self.ema_y:
            ema_yw = self.ema_net_y(out_y)
            y_hat = ema_yw * batch['prev_outputs'] + (1 - ema_yw) * y_hat
            # print(f'mean of ema_yw: {ema_yw.mean()}, max of ema_yw: {ema_yw.max()}, min of ema_yw: {ema_yw.min()}')

        return y_hat

    def forward_second_blocks_x(self, batch, br):
        # get out_x 
        if self.recursive:
            br = torch.cat((br, batch['current_treatments']), dim=-1)
        out_x = None
        if self.predict_X:
            if self.second_net == 'lstm':
                out_x, _ = self.hidden_net_x(br)
            elif self.second_net == 'tcn':
                out_x = self.hidden_net_x(br.transpose(1, 2)).transpose(1, 2)
            else:
                raise ValueError('second_net should be one of lstm and tcn')
            if not self.recursive:
                out_x = torch.cat((out_x, batch['current_treatments']), dim=-1)
        return out_x

    def forward_second_blocks_y(self, batch, br):
        # get out_y using the second net
        if self.recursive:
            br = torch.cat((br, batch['current_treatments']), dim=-1)
        if self.second_net == 'lstm':
            out_y, _ = self.hidden_net_y(br)
        elif self.second_net == 'tcn':
            out_y = self.hidden_net_y(br.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError('second_net should be one of lstm and tcn')
        if not self.recursive:
            out_y = torch.cat((out_y, batch['current_treatments']), dim=-1)
        return out_y


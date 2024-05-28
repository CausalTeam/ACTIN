import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, step=3, init=0.001):
        super(TemporalConvNet, self).__init__()
        self.step = step
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, init=init)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # return self.network(x)
        outputs = [x]  # store initial input in the outputs list
        for i, layer in enumerate(self.network):
            x = layer(x)
            outputs.append(x)
            # Check if the dimensions match for a skip connection and it's a step layer
            if i >= self.step and outputs[i-self.step].shape == x.shape:
                # Add skip connection
                x = x + outputs[i-self.step]
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, init=0.001):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 =Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.elu1, self.dropout1,
                                 self.conv2, self.chomp2, self.elu2, self.dropout2)
        
        self.downsample = Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.init = init
        
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, self.init)
        self.conv2.weight.data.normal_(0, self.init)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, self.init)

    def forward(self, x):
        # out = self.net(x)
        out = self.chomp1(self.conv1(x))
        out = self.dropout1(self.elu1(out))
        out = self.chomp2(self.conv2(out))
        out = self.dropout2(self.elu2(out))

        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        batch_size, _, seq_len = x.size()
        out = custom_conv1d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation) # (batch_size, output_channel, seq_len)
        return out

def custom_conv1d(x, weights, bias=None, stride=1, padding=0, dilation=1):
    batch_size, hdim, seq_len = x.size()
    out_channels, in_channels, kernel_size = weights.size()
    
    # add padding
    if padding > 0:
        x = F.pad(x, (padding, padding))
        # x = F.pad(x, (padding, padding))
        
    x_unfold = x.unfold(2, dilation*(kernel_size-1)+1, stride)[..., ::dilation] # (batch, input_channel, seq_len, kernel_size)
    output = torch.einsum('bisk,oik->bos', x_unfold, weights)
    if bias is not None:
        output = output + bias.unsqueeze(0).unsqueeze(-1).expand_as(output)

    return output

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        else:
            return x
import torch 
import torch.nn as nn
import torch.nn.functional as F  

from model.TinyHAR import BidLSTM, temporal_GRU, temporal_LSTM

temporal = {
    "bidlstm": BidLSTM,
    "lstm": temporal_LSTM,
    "gru": temporal_GRU
}

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, num_conv_layers, temporal_module, num_temp_layers, hidden_dim, kernel_size, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        
        # Conv
        self.conv_blocks = nn.ModuleDict({
            modality: self._conv_block(num_conv_layers, in_channel, hidden_dim, kernel_size) 
            for modality, in_channel in [('l_cap',4), ('r_cap',4), ('l_acc',3), ('r_acc',3), 
                                         ('l_gyro',3), ('r_gyro',3), ('l_quat',4), ('r_quat',4)]
            })
        
        # Temporal 
        self.temp_blocks = nn.ModuleDict({
            modality: temporal[temporal_module](hidden_dim, hidden_dim, num_temp_layers)
            for modality, num_channel in [('l_cap',4), ('r_cap',4), ('l_acc',3), ('r_acc',3), 
                                          ('l_gyro',3), ('r_gyro',3), ('l_quat',4), ('r_quat',4)]
            })

        self.fc1 = nn.Linear(hidden_dim * 8, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def _conv_block(self, num_conv_layers, in_channel, out_channel, kernel_size):
        layers_conv = []
        for i in range(num_conv_layers):
            if i%2 == 1:
                layers_conv.append(nn.Sequential(
                    nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=2),
                    nn.ReLU(inplace=True),#))#,
                    nn.BatchNorm1d(out_channel)))
            else:
                layers_conv.append(nn.Sequential(
                    nn.Conv1d(in_channel if i == 0 else out_channel, out_channel, kernel_size=kernel_size, stride=1),# (1,1)
                    nn.ReLU(inplace=True),#))#,
                    nn.BatchNorm1d(out_channel)))
                
        return nn.ModuleList(layers_conv)
    
    # def _temp_block(self, module, num_temp_layers, nb_channels, num_filters, hidden_dim):
    #     lstm_layers = []
    #     for i in range(num_temp_layers):
    #         lstm_layers.append(module(hidden_dim, hidden_dim))
    #     return nn.ModuleList(lstm_layers)
        
    def forward(self, x, device):
        
        # Conv1D per modality
        conv_outputs = {}
        
        for name, conv_block in self.conv_blocks.items():
            input_data = x[name].to(device).transpose(1,2)    # (B, L, C)

            for layer in conv_block:
                input_data = layer(input_data)                # (B, C, L)

            conv_outputs[name] = input_data.transpose(1,2)    # (B, C, L)

        # RNN per modality
        temp_hidden_states = []
        for name, layer in self.temp_blocks.items():
            # output from the previous conv step
            temp_input = conv_outputs[name]

            temp_input, hidden = layer(temp_input)
            
            temp_hidden_states.append(hidden[-1])
            
        concat_output = torch.cat(temp_hidden_states, dim=-1)
                
        out = self.relu(self.fc1(concat_output))
        out = self.fc2(self.dropout(out))
        
        return out


class Weighted_Sum_Late_Fusion(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Weighted_Sum_Late_Fusion, self).__init__()
        
        self.w_omega = nn.Parameter(torch.randn(hidden_size, attention_size) * 0.1)
        self.b_omega = nn.Parameter(torch.randn(attention_size) * 0.1)
        self.u_omega = nn.Parameter(torch.randn(attention_size) * 0.1)
        self.attention_size = attention_size
        self.hidden_size = hidden_size

    def forward(self, inputs, return_alphas=False):
        """
        inputs:
            - RNN outputs
            - Shape: (B, L, C) 
        """
        # PyTorch handles tuples from Bi-RNNs naturally by stacking. So don't need a specific check.

        # (B, L, C) @ (C, A) -> (B, L, A)
        v = torch.tanh(torch.tensordot(inputs, self.w_omega, dims=([2], [0])) + self.b_omega)
        
        # (B, L, A) @ (A) -> (B, L)
        vu = torch.tensordot(v, self.u_omega, dims=([2], [0]))
        # Softmax over the time dimension
        alphas = F.softmax(vu, dim=1) 

        # (B, L, 1) * (B, L, C) -> (B, L, C)
        # Summing over the time dimension to get (B, D)
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)

        if not return_alphas:
            return output
        else:
            return output, alphas

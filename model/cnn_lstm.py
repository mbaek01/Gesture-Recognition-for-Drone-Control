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
            modality: self._conv_block(num_conv_layers, hidden_dim, kernel_size) 
            for modality in ['l_cap', 'r_cap', 'l_acc', 'r_acc', 'l_gyro', 'r_gyro', 'l_quat', 'r_quat']
            })
        
        # Temporal 
        self.temp_blocks = nn.ModuleDict({
            modality: self._temp_block(temporal[temporal_module], num_temp_layers, num_channel, hidden_dim, hidden_dim)
            for modality, num_channel in [('l_cap',4), ('r_cap',4), ('l_acc',3), ('r_acc',3), ('l_gyro',3), ('r_gyro',3), ('l_quat',4), ('r_quat',4)]
            })

        self.fc1 = nn.Linear(hidden_dim * 8, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def _conv_block(self, num_conv_layers, filter_num, kernel_size):
        filter_num_list=[1]
        # filter_num_step=int(filter_num/num_conv_layers)
        for i in range(num_conv_layers-1):
            filter_num_list.append(filter_num)
        filter_num_list.append(filter_num)

        layers_conv = []
        for i in range(num_conv_layers):
            in_channel  = filter_num_list[i]
            out_channel = filter_num_list[i+1]
            if i%2 == 1:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (kernel_size, 1),(2,1)),
                    nn.ReLU(inplace=True),#))#,
                    nn.BatchNorm2d(out_channel)))
            else:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (kernel_size, 1),(1,1)),# (1,1)
                    nn.ReLU(inplace=True),#))#,
                    nn.BatchNorm2d(out_channel)))
                
        return nn.ModuleList(layers_conv)
    
    def _temp_block(self, module, num_temp_layers, nb_channels, num_filters, hidden_dim):
        lstm_layers = []
        for i in range(num_temp_layers):
            if i == 0:
                lstm_layers.append(module(nb_channels * num_filters, hidden_dim))
            else:
                lstm_layers.append(module(hidden_dim, hidden_dim))
        return nn.ModuleList(lstm_layers)
        
    def forward(self, x, device):
        conv_outputs = {}
        
        for name, conv_block in self.conv_blocks.items():
            input_data = x[name].to(device).unsqueeze(1)                                            # (B, f, L, C)

            for layer in conv_block:
                input_data = layer(input_data)

            conv_outputs[name] = torch.flatten(input_data.permute(0,2,3,1), start_dim=2, end_dim=3) # (B, L, C*f)

        temp_hidden_states = []
        for name, temp_block in self.temp_blocks.items():
            # output from the previous conv step
            temp_input = conv_outputs[name]

            for layer in temp_block:
                temp_input, hidden = layer(temp_input)
            
            temp_hidden_states.append(hidden[-1])
            
        concat_output = torch.cat(temp_hidden_states, dim=-1)
                
        out = self.relu(self.fc1(concat_output))
        out = self.fc2(self.dropout(out))
        
        return out





# class Weighted_Sum_Late_Fusion(nn.Module):

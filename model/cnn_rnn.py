import torch 
import torch.nn as nn
import torch.nn.functional as F  

from model.temporal_fusion import BidLSTM, temporal_GRU, temporal_LSTM, Temporal_Weighted_Aggregation

temporal = {
    "bidlstm": BidLSTM,
    "lstm": temporal_LSTM,
    "gru": temporal_GRU
}

# modality_aggregation = {
#     "fc": FC,
#     "attn": SelfAttention,
#     "attn_gamma": SelfAttention_Gamma
# }


class CNN_RNN(nn.Module):
    def __init__(self, 
                 modalities,
                 num_conv_layers, 
                 temporal_module, 
                 num_temp_layers, 
                 temp_agg, 
                 hidden_dim, 
                 kernel_size):
        super().__init__()

        self.modalities = modalities
                        # [('l_cap',4), ('r_cap',4), ('l_acc',3), ('r_acc',3), 
                        #    ('l_gyro',3), ('r_gyro',3), ('l_quat',4), ('r_quat',4)]

        self.hidden_dim = hidden_dim
        
        # Conv
        self.conv_blocks = nn.ModuleDict({
            modality: self._conv_block(num_conv_layers, in_channel, hidden_dim, kernel_size) 
            for modality, in_channel in self.modalities})
        
        # Temporal 
        self.temp_blocks = nn.ModuleDict({
            modality: temporal[temporal_module](hidden_dim, hidden_dim, num_temp_layers)
            for modality, _ in self.modalities})
        
        # Fusion
        if temporal_module == "bidlstm":
            self.hidden_dim *= 2
        
        if temp_agg:
            self.temp_agg = Temporal_Weighted_Aggregation(self.hidden_dim)
        
        else:
            self.temp_agg = None

        # self.modality_fusion = modality_aggregation[fusion_method](self.hidden_dim)

        # self.fc = nn.Linear(self.hidden_dim, num_classes)

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
        
        for name, _ in self.modalities:
            conv_block = self.conv_blocks[name]
            modality_input = x[name].to(device).transpose(1,2)  # (B, L, C)

            for layer in conv_block:
                modality_input = layer(modality_input)
            
            conv_outputs[name] = modality_input.transpose(1,2) # (B, C, L)
            
        # RNN per modality
        temp_hidden_states = [] # same order of modality as self.modalities

        for name, _ in self.modalities:
            # temp layer
            layer = self.temp_blocks[name]

            # input
            modality_input = conv_outputs[name]

            # output
            temp_output = layer(modality_input)

            if self.temp_agg:
                temp_output = self.temp_agg(temp_output)
            else:
                temp_output = temp_output[:, -1, :].squeeze(1)
            
            temp_hidden_states.append(temp_output)

        concat_output = torch.stack(temp_hidden_states, dim=1)

        return concat_output


import torch 
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, num_channels):
        super(SelfAttention).__init__()

        self.query = nn.Linear(num_channels, num_channels, bias=False)
        self.key   = nn.Linear(num_channels, num_channels, bias=False)
        self.value = nn.Linear(num_channels, num_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

        # TODO: test with and without
        self.fc1           = nn.Linear(num_channels, num_channels, bias=False)
        self.fc_activation = nn.ReLU() 
        self.fc2           = nn.Linear(num_channels, num_channels, bias=False)
        self.beta          = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        
        beta = F.softmax(torch.bmm(q, k.permute(0, 2, 1).contiguous()), dim=1)

        o = self.gamma * torch.bmm(v.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
        o = o.permute(0, 2, 1).contiguous()

        o = self.beta * self.fc2(self.fc_activation(self.fc1(o))) + o

        return o


class FilterWeighted_Aggregation(nn.Module):
    """

    """
    def __init__(self, n_channels):
        super(FilterWeighted_Aggregation, self).__init__()
        self.value_projection = nn.Linear(n_channels, n_channels)
        self.value_activation = nn.ReLU() 
        
        self.weight_projection = nn.Linear(n_channels, n_channels)
        self.weighs_activation = nn.Tanh() 
        self.softmatx = nn.Softmax(dim=1)        
        #self.fc            = nn.Linear(n_channels, n_channels)
        #self.fc_activation = nn.ReLU() 
        
    def forward(self, x):
        
        # batch  sensor_channel feature_dim


        weights = self.weighs_activation(self.weight_projection(x))
        weights = self.softmatx(weights)
        
        values  = self.value_activation(self.value_projection(x))

        values  = torch.mul(values, weights)
        o       = torch.sum(values,dim=1)


        o       = self.fc_activation(self.fc(o))
        # batch feature_dim
        return o
    
    
class temporal_GRU(nn.Module):
    """

    """
    def __init__(self, input_dim, output_dim, num_temp_layers=1):
        super(temporal_GRU, self).__init__()
        self.rnn = nn.GRU(input_dim,
                          output_dim,
                          num_layers = num_temp_layers,
                          bidirectional=False,
                          batch_first = True
                          )
    def forward(self, x):
        # Batch length Filter
        outputs, h = self.rnn(x)
        return outputs, h


class temporal_LSTM(nn.Module):
    """

    """
    def __init__(self, input_dim, output_dim, num_temp_layers=1):
        super(temporal_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim,  
                            num_layers = num_temp_layers,
                            batch_first = True)
    def forward(self, x):
        # Batch length Filter
        outputs, h = self.lstm(x)
        return outputs, h
    
class BidLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, num_temp_layers=1):
        super(temporal_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, 
                            num_layers = num_temp_layers,
                            batch_first = True,
                            bidirectional = True)
    def forward(self, x):
        # Batch length Filter
        outputs, h = self.lstm(x)
        return outputs, h



class FC(nn.Module):

    def __init__(self, channel_in, channel_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(channel_in ,channel_out)

    def forward(self, x):
        x = self.fc(x)
        return(x)


# class TinyHAR(nn.Module):
#     def __init__(self, num_channels, num_classes, num_conv_layers, num_lstm_layers, hidden_dim, dropout):
#         super(TinyHAR).__init__()

#         self.hidden_dim = hidden_dim

#         self.conv_lg_cap  = Conv_Block(4, hidden_dim, num_conv_layers)
#         self.conv_rg_cap  = Conv_Block(4, hidden_dim, num_conv_layers)

#         self.conv_lw_acc  = Conv_Block(3, hidden_dim, num_conv_layers)
#         self.conv_rw_acc  = Conv_Block(3, hidden_dim, num_conv_layers)

#         self.conv_lw_gyro = Conv_Block(3, hidden_dim, num_conv_layers)
#         self.conv_rw_gyro = Conv_Block(3, hidden_dim, num_conv_layers)

#         self.conv_lw_quat = Conv_Block(4, hidden_dim, num_conv_layers)
#         self.conv_rw_quat = Conv_Block(4, hidden_dim, num_conv_layers)

#         self.channel_interaction = SelfAttention(hidden_dim)

#         self.lstm_watch_l = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
#         self.lstm_watch_r = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
#         self.lstm_glove_l = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
#         self.lstm_glove_r = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)

#         self.channel_interaction = SelfAttention(hidden_dim)

#         self.channel_fusion = FilterWeighted_Aggregation(hidden_dim)

#         self.temporal_GRU = temporal_GRU(hidden_dim)

#         self.FC = FC(hidden_dim, num_classes)

#     def forward(self, x):
#         x = x.unsqueeze(1)

#         l_cap, r_cap = x['l_cap'], x['r_cap']

#         l_acc, r_acc = x['l_acc'], x['r_acc']
#         l_gyro, r_gyro = x['l_gyro'], x['r_gyro']
#         l_quat, r_quat = x['l_quat'], x['r_quat']

#         conv_l_cap = self.conv_lg_cap(l_cap)
#         conv_r_cap = self.conv_rg_cap(r_cap)

#         conv_l_acc = self.conv_lw_acc(l_acc)
#         conv_r_acc = self.conv_rw_acc(r_acc)

#         conv_l_gyro = self.conv_lw_gyro(l_gyro)
#         conv_r_gyro = self.conv_rw_gyro(r_gyro)

#         conv_l_quat = self.conv_lw_quat(l_quat)
#         conv_r_quat = self.conv_rw_quat(r_quat)

#         conv_concat = torch.cat([conv_l_cap, conv_r_cap,
#                                  conv_l_acc, conv_r_acc, 
#                                  conv_l_gyro, conv_r_gyro,
#                                  conv_l_quat, conv_r_quat], dim=-1)

#         cross_channel = torch.cat([self.channel_interaction(conv_concat[:, :, t, :]).unsqueeze(3) 
#                                    for t in range(conv_concat.shape[2])], dim=-1)
        
#         temporal = self.temporal_GRU(cross_channel)

#         final = FC(temporal)

#         return final

        
    

        
        
        



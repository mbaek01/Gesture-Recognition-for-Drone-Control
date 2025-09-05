import torch 
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention_Gamma(nn.Module):
    """

    """
    def __init__(self, n_channels):
        super(SelfAttention_Gamma, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key   = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

        self.fc = nn.Linear(n_channels*8, n_channels)

    def forward(self, x):

        f, g, h = self.query(x), self.key(x), self.value(x)
        
        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)

        o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) # + x.permute(0, 2, 1).contiguous()
        o = o.permute(0, 2, 1).contiguous()

        # for attention weight visualization
        scaled_weights = self.gamma*beta

        o = torch.flatten(o, 1, 2)
        o = self.fc(o)

        return o, scaled_weights

class SelfAttention(nn.Module):
    """

    """
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key   = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)

        self.fc = nn.Linear(n_channels*8, n_channels)

    def forward(self, x):

        f, g, h = self.query(x), self.key(x), self.value(x)
        
        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)

        o = torch.bmm(h.permute(0, 2, 1).contiguous(), beta) # + x.permute(0, 2, 1).contiguous()
        o = o.permute(0, 2, 1).contiguous()

        o = torch.flatten(o, 1, 2)
        o = self.fc(o)

        return o, beta


class Temporal_Weighted_Aggregation(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(Temporal_Weighted_Aggregation, self).__init__()

        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.weighs_activation = nn.Tanh() 
        self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
        self.sm = torch.nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # x: output from RNN

        out = self.weighs_activation(self.fc_1(x))

        out = self.fc_2(out).squeeze(2)

        weights_att = self.sm(out).unsqueeze(2)

        context = torch.sum(weights_att * x, 1)
        context = x[:, -1, :] + self.gamma * context
        return context
    

class FC(nn.Module):

    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(8*hidden_dim ,hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1, 2)
        x = self.relu(self.fc(x))
        return x


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
        return outputs


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
        return outputs
    

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
        return outputs



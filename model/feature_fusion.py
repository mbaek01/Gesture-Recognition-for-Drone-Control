import torch.nn as nn

from model.cnn_rnn import CNN_RNN
from model.temporal_fusion import SelfAttention, SelfAttention_Gamma, FC

modality_aggregation = {
    "fc": FC,
    "attn": SelfAttention,
    "attn_gamma": SelfAttention_Gamma
}


class Feature_Fusion(nn.Module):
    """
    Model for feature-level fusion, built on top of the CNN_RNN feature backbone.
    """
    def __init__(self, 
                 num_classes, 
                 num_conv_layers, 
                 temporal_module, 
                 num_temp_layers, 
                 temp_agg, 
                 fusion_method,
                 hidden_dim, 
                 kernel_size):
        super().__init__()
        
        self.feature_extractor = CNN_RNN(num_conv_layers, 
                                        temporal_module, 
                                        num_temp_layers, 
                                        temp_agg, 
                                        hidden_dim, 
                                        kernel_size)
        
        self.modalities = self.feature_extractor.modalities
                                        
        self.hidden_dim = hidden_dim

        if temporal_module == "bidlstm":
            self.hidden_dim *= 2

        # modality method 
        self.modality_fusion = modality_aggregation[fusion_method](self.hidden_dim)

        # final classification head 
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        concat_x = self.feature_extractor(x)

        all_modalities, att_weights = self.modality_fusion(concat_x)
        
        out = self.fc(all_modalities)
        
        return out, att_weights
    
    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
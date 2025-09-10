import torch
import torch.nn as nn

from .cnn_rnn import CNN_RNN

class LLR_Fusion(nn.Module):
    """
    Model for decision-level fusion. It outputs separate logits
    for each modality.
    """
    def __init__(self, 
                 modalities,
                 num_classes, 
                 num_conv_layers, 
                 temporal_module, 
                 num_temp_layers, 
                 temp_agg,
                 hidden_dim, 
                 kernel_size):
        super().__init__()
        
        self.feature_extractor = CNN_RNN(modalities,
                                        num_conv_layers, 
                                        temporal_module, 
                                        num_temp_layers, 
                                        temp_agg, 
                                        hidden_dim, 
                                        kernel_size
                                        ) 
                                        
        self.modalities = self.feature_extractor.modalities
        # self.modalities = [('l_cap',4), # 0
        #                    ('r_cap',4), # 1
        #                    ('l_acc',3), # 2 
        #                    ('r_acc',3), # 3
        #                    ('l_gyro',3),# 4 
        #                    ('r_gyro',3),# 5 
        #                    ('l_quat',4),# 6 
        #                    ('r_quat',4)]# 7

        if temporal_module == "bidlstm":
            self.hidden_dim = hidden_dim * 2
        else:
            self.hidden_dim = hidden_dim
        
        # A separate classification head for each modality
        self.classification_heads = nn.ModuleDict({
            modality: nn.Linear(self.hidden_dim, num_classes)
            for modality, _ in self.modalities
        })

    def forward(self, x, device):
        # The backbone returns a single concatenated tensor of all modalities
        concat_x = self.feature_extractor(x, device) # (B, num_modalities, hidden_dim)
        
        # classification head -> LLR
        per_modality_llrs = {}

        for i, (name, _) in enumerate(self.modalities):
            # Extract the hidden state for the current modality
            hidden_state = concat_x[:, i, :]                        # (B, hidden_dim)
            
            # Pass to its classification head
            logits = self.classification_heads[name](hidden_state)  # (B, num_classes)
            llrs = calculate_all_llrs(logits)                       # (B, num_classes)
            per_modality_llrs[name] = llrs

        summed_llr = torch.sum(torch.stack(list(per_modality_llrs.values())), dim=0)

        # Return the dictionary for easy visualization
        return summed_llr, per_modality_llrs


def calculate_all_llrs(logits):
    """
    Calculates the Log-Likelihood Ratio ln(p(c) / p(not c)) for all classes.

    Args:
        logits (torch.Tensor): A tensor of shape (batch_size, num_classes).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_classes) containing the LLRs.
    """
    # Get the log-sum-exp over all classes. This is a stable way to get the
    # log of the denominator of the softmax.
    # keepdim=True is important for broadcasting. Shape: (B, 1)
    lse_all = torch.logsumexp(logits, dim=1, keepdim=True)

    # We need log(sum(exp(z_i))) for all i != c.
    # This can be computed as log( sum(exp(z_i))_all - exp(z_c) )
    # This is equivalent to log( exp(lse_all) - exp(z_c) )
    lse_others = torch.log(torch.exp(lse_all) - torch.exp(logits))

    # LLR_c = z_c - lse_others_c
    llrs = logits - lse_others
    return llrs


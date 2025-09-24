import torch.nn as nn
import torch
from thop import profile
import os

from model.feature_fusion import Feature_Fusion
from model.llr_fusion import LLR_Fusion

def get_model(args, filtered_modalities):

    if args.model == "feature_fusion":
        model = Feature_Fusion(filtered_modalities,
                                args.num_classes, 
                                args.num_conv_layers, 
                                args.temporal_module,
                                args.num_temp_layers, 
                                args.temp_agg,
                                args.fusion_method,
                                args.hidden_dim, 
                                args.kernel_size
                                )
            
    elif args.model == "llr_fusion":
        model = LLR_Fusion(filtered_modalities,
                            args.num_classes, 
                            args.num_conv_layers, 
                            args.temporal_module,
                            args.num_temp_layers, 
                            args.temp_agg,
                            args.hidden_dim, 
                            args.kernel_size
                            )
    
    else:
        raise NotImplementedError

    return model


def get_model_size(model: nn.Module):
    """Calculates the model size in megabytes (MB)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def get_model_profile(model: nn.Module, modalities, batch_size, device, logger, name, path):
    dummy_input = {m: torch.rand(batch_size, 100, channel) for m,channel in modalities}

    macs, _ = profile(model, inputs=(dummy_input, device))
    model_size = get_model_size(model)


    # Convert MACs to GFLOPs (1 GFLOPs ≈ 2 GMACs)
    gflops = (macs * 2) / 1e9

    # log
    profile_str = f"[{name}]\n GFLOPs: {gflops:.2f} \n Model Size: {model_size:.2f} MB \n"

    profile_log = open(os.path.join(path, "model_info.txt"), "a")

    profile_log.write(profile_str)
    profile_log.flush()

    
    logger.info(profile_str)


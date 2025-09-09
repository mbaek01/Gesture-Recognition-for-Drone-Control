from model.feature_fusion import Feature_Fusion
from model.llr_fusion import LLR_Fusion


def get_model(args):
    
    if args.model == "feature_fusion":
        model = Feature_Fusion(args.num_classes, 
                                args.num_conv_layers, 
                                args.temporal_module,
                                args.num_temp_layers, 
                                args.temp_agg,
                                args.fusion_method,
                                args.hidden_dim, 
                                args.kernel_size
                                )
            
    elif args.model == "llr_fusion":
        model = LLR_Fusion(args.num_classes, 
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

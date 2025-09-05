from model.cnn_rnn import CNN_RNN


def get_model(args):
    
    if args.model == "cnn_rnn":
        model = CNN_RNN(args.num_classes, 
                        args.num_conv_layers, 
                        args.temporal_module,
                        args.num_temp_layers, 
                        args.temp_agg,
                        args.fusion_method,
                        args.hidden_dim, 
                        args.kernel_size
                        )
    
    else:
        raise NotImplementedError

    return model

from model.cnn_lstm import CNN_LSTM


def get_model(args):
    
    if args.model == "cnn_lstm":
        model = CNN_LSTM(args.num_classes, 
                         args.num_conv_layers, 
                         args.temporal_module,
                         args.num_temp_layers, 
                         args.hidden_dim, 
                         args.kernel_size,
                         args.dropout)
    
    else:
        raise NotImplementedError

    return model

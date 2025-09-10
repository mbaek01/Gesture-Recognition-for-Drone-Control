import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def get_args():
    parser = argparse.ArgumentParser(description="Gesture Recognition for Drone Control")

    # Dataset-related arguments
    parser.add_argument('--model', type=str, default='llr_fusion',
                        help='model name', choices=['llr_fusion', 'feature_fusion'])
    # parser.add_argument('--dataset_version', type=str, default='v1',
    #                     help='model version: v1, v3 only valid for model version v1 and v1-improved')
    parser.add_argument('--normalize', type=lambda x: x.lower() == 'true', default=True,
                        help="normalize data before training 'true' or 'false'., default is true")
    
    parser.add_argument('--dataset_path', type=str, default='/workspace/drone_gesture/full_dataset',
                        help='Path to the dataset')
    parser.add_argument('--save_path', type=str, default='/workspace/drone_gesture/saved',
                        help='Path to the dataset')
    
    parser.add_argument('--sliding_window_size', type=int, default=3,
                        help='Sliding window size')
    parser.add_argument('--sliding_window_step', type=int, default=1,
                        help='Sliding window step size')
    parser.add_argument('--skip_null_class', type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to skip null class. Pass 'true' or 'false'.")
    
    # Training-related arguments
    parser.add_argument('--seed', type=int, default=42, help="random seed number")
    parser.add_argument('--gpu', default=0, type=int, help="gpu index number")

    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loaders')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='Learning rate for the optimizer')
    
    parser.add_argument('--loso', type=bool, default=str2bool, help="perform loso experiments")
    parser.add_argument('--lopo', type=bool, default=str2bool, help="perform lopo experiments")
    parser.add_argument('--train_valid_split_ratio', type=float, default=0.9, help="ratio of train dataset to valid dataset")
    
    # Model-related arguments
    parser.add_argument('--num_conv_layers', type=int, default=4,
                        help='Number of convolutional layers')
    parser.add_argument('--kernel_size', type=int,
                        default=5, help='Convolution kernel size, (k, 1)')
    
    parser.add_argument('--temporal_module', type=str,
                    default="gru", help='Type of temporal module', choices=['bidlstm', 'lstm', 'gru'])
    parser.add_argument('--num_temp_layers', type=int,
                        default=2, help='Number of temporal module layers')
    
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden size for CNN and LSTM layers')
    # parser.add_argument('--dropout', type=float, default=0.3,
    #                     help='Dropout rate for the model')
    parser.add_argument('--num_classes', type=int,
                        default=20, help='Number of output classes')
    
    # modalities to use in training
    parser.add_argument('--modalities', 
                        nargs='+',  # This is the key: accepts 1 or more arguments
                        default=['l_cap', 'r_cap', 'l_acc', 'r_acc','l_gyro', 'r_gyro', 'l_quat', 'r_quat'],
                        help='A space-separated list of modalities to use.')
    
    parser.add_argument('--temp_agg', type=str2bool, default=True, 
                            help="Whether to use weighted sum of rnn's hidden state")

    # for args.model == feature_fusion only
    parser.add_argument('--fusion_method', type=str, default='attn',
                        help="Modality fusion method", choices=['fc', 'attn', 'attn_gamma'])
    
    args = parser.parse_args()

    if not args.skip_null_class:
        assert args.num_classes == 21, "Skipping null class should had a total of 21 classes"
    
    else: 
        assert args.num_classes < 21, "Skipping null class should had less than 21 classes"
    

    return args

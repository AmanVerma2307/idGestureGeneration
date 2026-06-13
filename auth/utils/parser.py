import argparse

def parse():
    
    parser = argparse.ArgumentParser()

    #### General arguments
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed to ensure reproducibility")
    parser.add_argument('--multi_gpu',
                        type=int,
                        default=0,
                        help="If True, then multi GPU training/testing")
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help="Device to be used in single gpu setting")

    #### Dataset arguments
    parser.add_argument('--dataset',
                        type=str,
                        default='scut',
                        help="Dataset to be used for train/test: [soli,tiny,handLogin,scut]")
    parser.add_argument('--valSplit',
                        type=float,
                        default=0.2,
                        help="Size of validation set wrt Train set")
    parser.add_argument('--sample',
                        type=int,
                        default=0,
                        help="If True then sampling over frames will be performed")
    parser.add_argument('--sampleMethod',
                        type=str,
                        default='sample',
                        help="Sampling method: [sample/continuous/random], default:sample")
    parser.add_argument('--numFrames',
                        type=int,
                        default=64,
                        help="Number of frames to be sampled")
    
    #### Training arguments
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="Batch size to be used in training")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help="Learning rate to be used in training")
    parser.add_argument('--exp_name',
                        type=str,
                        help="Name of the experiment")
    parser.add_argument('--num_epochs',
                        type=int,
                        default=50,
                        help="Number of training epochs")
    
    #### Model arguments
    parser.add_argument('--modelChoice',
                        type=str,
                        default='vivit',
                        help="Model to be used")
    
    ### ViViT
    parser.add_argument('--patchSizeT',
                        type=int,
                        default=5,
                        help="Patch size to be used along T dimension")
    parser.add_argument('--patchSizeH',
                        type=int,
                        default=20,
                        help="Patch size to be used along H dimension")
    parser.add_argument('--patchSizeW',
                        type=int,
                        default=20,
                        help="Patch size to be used along W dimension")
    
    parser.add_argument('--numEncoders',
                        type=int,
                        default=2,
                        help="Number of encoders to be used")
    parser.add_argument('--numHeads',
                        type=int,
                        default=8,
                        help="Number of attention heads to be used")
    parser.add_argument('--input_dim',
                        type=int,
                        default=32,
                        help="Input number of channels to tubelet embedding layer")
    parser.add_argument('--d_model',
                        type=int,
                        default=32,
                        help="Embedding dimensions")
    parser.add_argument('--dff',
                        type=int,
                        default=128,
                        help="Embedding hidden dimensions")
    parser.add_argument('--vivit_rate',
                        type=float,
                        default=0.2,
                        help="ViViT dropout rate")
    
    args = parser.parse_args()
    return args
    
    
    
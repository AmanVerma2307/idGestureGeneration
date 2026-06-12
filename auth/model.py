import torch
from auth.models.vivit import *
from auth.models.msba import *

def getModel(args,
             T,
             H,
             W,
             C):
    """
    Function to get model
    """

    if(args.modelChoice == 'vivit'):
        model = res3dViViT(input_dim=args.d_model,
                           patch_size=[args.pathSizeT,args.patchSizeH,args.patchSizeW],
                           T=T,
                           H=H,
                           W=W,
                           C=C,
                           d_model=args.d_model,
                           num_heads=args.num_heads,
                           dff=2*args.d_model,
                           rate=args.vivit_rate,
                           num_encoders=args.numEncoders)
        
    if(args.modelChoice == 'msba'):
        model = Model_MSBANet(frame_length=T,
                              feature_dim=512,
                              out_dim=128)

    return model
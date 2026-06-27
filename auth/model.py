import torch
from models.vivit import *
from models.msba import *
from models.resnet import *
from models.i3d import *
from models.fstanet import *
from models.videomae import *

def getModel(args,
             T,
             H,
             W,
             C,
             I):
    """
    Function to get model
    """

    if(args.modelChoice == 'vivit'):
        model = res3dViViT(input_dim=args.input_dim,
                           patch_size=[args.patchSizeT,args.patchSizeH,args.patchSizeW],
                           T=T,
                           H=H,
                           W=W,
                           C=C,
                           d_model=args.d_model,
                           num_heads=args.numHeads,
                           dff=args.dff,
                           rate=args.vivit_rate,
                           num_encoders=args.numEncoders,
                           I=I)
        
    if(args.modelChoice == 'msba'):
        model = Model_MSBANet(frame_length=T,
                              feature_dim=512,
                              out_dim=128)
        
    if(args.modelChoice == 'resnet18'):
        model = resnet18Video(I)

    if(args.modelChoice == 'i3d'):
         model = i3d(I)

    if(args.modelChoice == 'fstanet'):
         model = fsta(T,I)

    if(args.modelChoice == 'videomae'):
         model = videomae(I,modelStyle="B")

    total_params = sum(p.numel() for p in model.parameters())
    print('++++++++++++++++++')
    print('Model: '+str(args.modelChoice))
    print(model)
    print('Total parameters: '+str(total_params)) 
    print('++++++++++++++++++')

    return model
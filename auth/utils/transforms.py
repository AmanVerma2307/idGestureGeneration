import torch
import torchvision
from torchvision.transforms import v2

def getTransforms(args):

    """
    Function to get transforms

    INPUTS:-
    1) args: Input arguments

    OUTPUTS:-
    1) tranform: A composed list of torchvision.transforms
    """

    if(args.applyTransforms == 1):
        if(args.transformMode == 'elastic'):
            transform = torchvision.transforms.v2.Compose([v2.ElasticTransform(alpha=args.elasticTx_alpha),
                                                           v2.CenterCrop((int(args.sizeH*0.85),int(args.sizeW*0.85))),
                                                           v2.Resize((args.sizeH,args.sizeW))
                                                          ])
            
        if(args.transformMode == 'blur'):
            transform = torchvision.transforms.v2.Compose([v2.GaussianBlur(kernel_size=(args.blur_kernelSize), 
                                                                           sigma=(args.blur_sigma))
                                                          ])
            
        if(args.transformMode == 'brightness'):
            transform = torchvision.transforms.v2.Compose([v2.ColorJitter(brightness=(0.1,1.0))
                                                          ])
            
        if(args.transformMode == 'colorJitter'):
            transform = torchvision.transforms.v2.Compose([v2.ColorJitter(brightness=(0.1,1.0),
                                                                          saturation=(0.1,1.0),
                                                                          hue=(0.1,0.4),
                                                                          contrast=(0, 10))
                                                          ])
            
        if(args.transformMode == 'elastic_and_blur'):
            transform = torchvision.transforms.v2.Compose([v2.ElasticTransform(alpha=args.elasticTx_alpha),
                                                           v2.CenterCrop((int(args.sizeH*0.85),int(args.sizeW*0.85))),
                                                           v2.Resize((args.sizeH,args.sizeW)),
                                                           v2.GaussianBlur(kernel_size=(args.blur_kernelSize), 
                                                                           sigma=(args.blur_sigma))
                                                          ])
            
        if(args.transformMode == 'blur_and_bright'):
            transform = torchvision.transforms.v2.Compose([v2.GaussianBlur(kernel_size=(args.blur_kernelSize), 
                                                                        sigma=(args.blur_sigma)),
                                                           v2.ColorJitter(brightness=(0.1,1.0))
                                                          ])
            
        if(args.transformMode == 'elastic_and_bright'):
            transform = torchvision.transforms.v2.Compose([v2.ElasticTransform(alpha=args.elasticTx_alpha),
                                                           v2.CenterCrop((int(args.sizeH*0.85),int(args.sizeW*0.85))),
                                                           v2.Resize((args.sizeH,args.sizeW)),
                                                           v2.ColorJitter(brightness=(0.1,1.0))
                                                          ])


        if(args.transformMode == 'combined'):
            transform = torchvision.transforms.v2.Compose([v2.ElasticTransform(alpha=args.elasticTx_alpha),
                                                           v2.CenterCrop((int(args.sizeH*0.85),int(args.sizeW*0.85))),
                                                           v2.Resize((args.sizeH,args.sizeW)),
                                                           v2.GaussianBlur(kernel_size=(args.blur_kernelSize), 
                                                                        sigma=(args.blur_sigma)),
                                                           v2.ColorJitter(brightness=(0.1,1.0))
                                                          ])
        
        return transform
    else:
        return None
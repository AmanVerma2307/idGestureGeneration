import wandb
import random
import torch
import numpy as np
from model import *
from epoch.epoch import trainVal
from utils.parser import *
from utils.transforms import *
from data.scut import *

args = parse()

if(args.reproducibility == 1):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if(args.trackMetrics == 1):
    wandb.init(project='idGestureAuth',
            name=args.exp_name)

if(args.multi_gpu == 0):
    device = torch.device(args.device)

if(args.dataset == 'scut'):
    trainDataset = scutDataset(mode='train',
                            splitSize=args.valSplit,
                            numFrames=args.numFrames,
                            sample=args.sample,
                            sampleMethod=args.sampleMethod,
                            H=args.sizeH,
                            W=args.sizeW,
                            transform=getTransforms(args),
                            motionModel=args.motionModel)
    valDataset = scutDataset(mode='val',
                            splitSize=args.valSplit,
                            numFrames=args.numFrames,
                            sample=args.sample,
                            sampleMethod=args.sampleMethod,
                            H=args.sizeH,
                            W=args.sizeW,
                            transform=getTransforms(args),
                            motionModel=args.motionModel)
    
    T = args.numFrames
    H = args.sizeH
    W = args.sizeW
    C = 3
    I = 143


trainLoader = torch.utils.data.DataLoader(trainDataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          drop_last=False,
                                          pin_memory=True)
valLoader = torch.utils.data.DataLoader(valDataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True)

if(args.motionModel == 0):
    model = getModel(args, 
                    T, 
                    H, 
                    W, 
                    C, 
                    I)
else:
    model = motionModel(args,I)
model.to(device)

criterion_id = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

wandb.watch(model,criterion_id,log="all",log_freq=1)

train_metrics, val_metrics = trainVal(trainLoader,
                                      valLoader,
                                      model,
                                      optimizer,
                                      criterion_id,
                                      args)

##### Saving
np.savez_compressed('./auth/_store/_history/'+args.exp_name+'_trainMetrics.npz',np.array(train_metrics))
np.savez_compressed('./auth/_store/_history/'+args.exp_name+'_valMetrics.npz',np.array(val_metrics))
import wandb
import random
import torch
import numpy as np
from auth.model import getModel
from auth.epoch import trainVal
from auth.utils.parser import *
from auth.data.scut import *

args = parse()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

wandb.init(project='idGestureGeneration',name=args.exp_name)

if(args.multi_gpu == False):
    device = torch.device(args.device)

trainDataset = scutDataset(mode='train',
                          splitSize=args.valSplit,
                          numFrames=args.numFrames,
                          sample=args.sample,
                          sampleMethod=args.sampleMethod)
valDataset = scutDataset(mode='val',
                         splitSize=args.valSplit,
                         numFrames=args.numFrames,
                         sample=args.sample,
                         sampleMethod=args.sampleMethod)


trainLoader = torch.utils.data.DataLoader(trainDataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          drop_last=False)
valLoader = torch.utils.data.DataLoader(valDataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=False)

if(args.dataset == 'scut'):
    input_dim = 32
    T = args.numFrames
    H = 200
    W = 200
    C = 3
    d_model = args.d_model

model = getModel(args, T, H, W, C, input_dim)
model.to(device)

criterion_id = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

wandb.watch(model,criterion_id,log="all",log_freq=1)

train_metrics, val_metrics = trainVal(trainLoader,
                                      valLoader,
                                      model,
                                      optimizer,
                                      criterion_id,
                                      args)

##### Saving
np.savez_compressed('./model history/'+args.exp_name+'_trainMetrics.npz',np.array(train_metrics))
np.savez_compressed('./model history/'+args.exp_name+'_valMetrics.npz',np.array(val_metrics))
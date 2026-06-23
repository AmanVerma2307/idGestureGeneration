import random
import torch
import numpy as np
from model import getModel
from utils.parser import *
from utils.eerComp import *
from data.scut import *

args = parse()

if(args.reproducibility == 1):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if(args.dataset == 'scut'):

    gallery = scutDataset(mode='test',
                          splitSize=args.valSplit,
                          numFrames=args.numFrames,
                          sample=args.sample,
                          sampleMethod=args.sampleMethod,
                          sessionID=1,
                          H=args.sizeH,
                          W=args.sizeW)
    galleryLabels, _ = gallery.__getLabels__()

    probe = scutDataset(mode='test',
                        splitSize=args.valSplit,
                        numFrames=args.numFrames,
                        sample=args.sample,
                        sampleMethod=args.sampleMethod,
                        sessionID=2,
                        H=args.sizeH,
                        W=args.sizeW)
    probeLabels, _ = probe.__getLabels__()
    
    T = args.numFrames
    H = args.sizeH
    W = args.sizeW
    C = 3
    I = 143

model = getModel(args, 
                 T, 
                 H, 
                 W, 
                 C, 
                 I)
model.load_state_dict(torch.load('./auth/_store/_weights/'+args.exp_name+'.pth', weights_only=True)) 
model.eval()

if(args.multi_gpu == 0):
    device = torch.device(args.device)
    model.to(device)


_, gallery = model.predict(torch.utils.data.DataLoader(gallery,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       pin_memory=True),
                                                       args)

_, probe = model.predict(torch.utils.data.DataLoader(probe,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     pin_memory=True),
                                                     args)

eerVal, bestThresh = calculate_eer(probe,
                                   gallery,
                                   probeLabels,
                                   galleryLabels,
                                   useCosine=1)

measure = ['model', 'eerVal', 'bestThresh']
measureVal = [str(args.exp_name),
              str(round(eerVal,4)),
              str(round(bestThresh,4))]

print('===============================')
print('EER Value: '+str(round(eerVal,4)))
print('Best threshold: '+str(round(bestThresh,4)))

if(args.resultFileInit == 1):
    resultFile = open('./auth/_store/_resultFiles/'+args.resultFileName+'.txt','w')

    for idx, item in enumerate(measure):
        if(idx == 0):
            resultFile.write(str(item)+'                                     ')
        if(idx == 1):
            resultFile.write(str(item)+'           ')
        if(idx == 2):
            resultFile.write(str(item)+'\n')

    for idx, item in enumerate(measureVal):
        if(idx == 0):
            resultFile.write(str(item)+'                                     ')
        if(idx == 1):
            resultFile.write(str(item)+'           ')
        if(idx == 2):
            resultFile.write(str(item)+'\n')

if(args.resultFileInit == 0):
    resultFile = open('./auth/_store/_resultFiles/'+args.resultFileName+'.txt','a')
    for idx, item in enumerate(measureVal):
        if(idx == 0):
            resultFile.write(str(item)+'                                     ')
        if(idx == 1):
            resultFile.write(str(item)+'           ')
        if(idx == 2):
            resultFile.write(str(item)+'\n')
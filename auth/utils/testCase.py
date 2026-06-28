import os
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
import skimage.io as skio
import matplotlib.pyplot as plt
from torchvision.io import decode_image
import PIL
from pathlib import Path
# from data.scut import *
from skimage.transform import resize


def processFrame(img_path, 
                 H=128, 
                 W=128,
                 transform = None):

    """
    Function to Read the Frame and Preprocess them.

    INPUTS:-
    1) img_path: Path to the image

    OUTPUTS:-
    1) frame: Preprocessed image of Dimensions - (3xHxW)
    """
    if(H != 200 and W != 200):
        frame = np.transpose(resize(skio.imread(img_path),(H,W)),(2,0,1))
    else:
        frame = np.transpose(skio.imread(img_path),(2,0,1))
    frame = (frame-np.min(frame,axis=(1,2),keepdims=True))/(np.max(frame,axis=(1,2),keepdims=True)-np.min(frame,axis=(1,2),keepdims=True))

    if(transform == None):
        return frame
    else:
        return transform(frame)


def processFolder(folderPath,
                  numFrames=64,
                  sample=0,
                  sampleMethod='continuous',
                  H=128,
                  W=128,
                  transform=None):

    """
    Function to process entire gesture sample stored inside a folder

    INPUTS:-
    1) folderPath: Path to the folder
    2) numFrames: Total number of frames in the input video (Default fixed @ 64 frames)
    3) sample: If 1 then sampling, else full video
    4) sampleMethod: Sampling method

    OUTPUTS:-
    1) gestTensor: Torch tensor of shape [3 x numFrames x 200 x 200]
    """

    gestTensor = []
    totalFrames = 64 # Total number of frames

    if(sample == 0):
        for frameIdx, frameName in enumerate(sorted(os.listdir(folderPath))):
            gestTensor.append(processFrame(folderPath+'/'+frameName,H=H,W=W))

    if(sample == 1):
        if(sampleMethod == 'continuous'):
            for frameIdx, frameName in enumerate(sorted(os.listdir(folderPath))):
                if((frameIdx+1)<=numFrames):
                    gestTensor.append(processFrame(folderPath+'/'+frameName,H=H,W=W))

        if(sampleMethod == 'sample'):
            frameNames = sorted(os.listdir(folderPath))
            frameNumbers = list(np.int32(np.round(np.linspace(0,totalFrames-1,numFrames))))

            for idx in frameNumbers:
                gestTensor.append(processFrame(folderPath+'/'+frameNames[idx],H=H,W=W))
                
        if(sampleMethod == 'random'):
            frameNames = sorted(os.listdir(folderPath))
            frameNumbers = sorted(list(np.random.randint(0,totalFrames,size=(numFrames,),)))

            for idx in frameNumbers:
                gestTensor.append(processFrame(folderPath+'/'+frameNames[idx],H=H,W=W))

        if(sampleMethod == 'randomOrder'):
            frameNames = sorted(os.listdir(folderPath))
            frameNumbers = list(np.random.randint(0,totalFrames,size=(numFrames,)))

            for idx in frameNumbers:
                gestTensor.append(processFrame(folderPath+'/'+frameNames[idx],H=H,W=W))

    if(transform == None):
        return torch.Tensor(np.transpose(np.array(gestTensor),(1,0,2,3)))
    else:
        return transform(torch.Tensor(gestTensor)).permute(1,0,2,3)

# a = np.random.normal(size=(10,32))
# b = np.random.normal(size=(10,64))

H = 128
W = 128

# for item1, item2 in zip(a,b):
#     print(item1.size, item2.size)
transform = torchvision.transforms.v2.Compose([
    #v2.ElasticTransform(alpha=500),
    #v2.CenterCrop((int(H*0.85),int(W*0.85))),
    #v2.Resize((H,W))
    v2.GaussianBlur(kernel_size=(11), sigma=(5)),
    # v2.ColorJitter(brightness=(0.2,0.5))
])

img = processFolder('./datasets/scut/scut/color_hand/1_1_132_0_3/', transform=transform)
#torch.Tensor(skio.imread('./datasets/scut/scut/color_hand/1_1_0_0_0/10.jpg')).permute(2,0,1)
#img = img/255
print(img.shape,type(img))
img = img.permute((1,0,2,3))
# print(img.shape)
# plt.imshow(b.numpy().transpose(1,2,0))
# plt.show()

for op in img:
    print(op.shape)
    plt.imshow(op.numpy().transpose((1,2,0)))
    plt.show()
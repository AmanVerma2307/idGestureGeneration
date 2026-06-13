import os
import torch
import numpy as np
import skimage.io as skio
from sklearn.utils import shuffle


def processFrame(img_path):

    """
    Function to Read the Frame and Preprocess them.

    INPUTS:-
    1) img_path: Path to the image

    OUTPUTS:-
    1) frame: Preprocessed image of Dimensions - (3x200x200)
    """
    frame = np.transpose(skio.imread(img_path),(2,0,1))
    return (frame-np.min(frame,axis=(1,2),keepdims=True))/(np.max(frame,axis=(1,2),keepdims=True)-np.min(frame,axis=(1,2),keepdims=True))


def processFolder(folderPath,
                  numFrames=64,
                  sample=0,
                  sampleMethod='continuous'):

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
            gestTensor.append(processFrame(folderPath+'/'+frameName))

    if(sample == 1):
        if(sampleMethod == 'continuous'):
            for frameIdx, frameName in enumerate(sorted(os.listdir(folderPath))):
                if((frameIdx+1)<=numFrames):
                    gestTensor.append(processFrame(folderPath+'/'+frameName))

        if(sampleMethod == 'sample'):
            frameNames = sorted(os.listdir(folderPath))
            frameNumbers = list(np.int32(np.round(np.linspace(0,totalFrames-1,numFrames))))

            for idx in frameNumbers:
                gestTensor.append(processFrame(folderPath+'/'+frameNames[idx]))
                
        if(sampleMethod == 'random'):
            frameNames = sorted(os.listdir(folderPath))
            frameNumbers = sorted(list(np.random.randint(0,totalFrames,size=(numFrames,))))

            for idx in frameNumbers:
                gestTensor.append(processFrame(folderPath+'/'+frameNames[idx]))

    return torch.Tensor(np.transpose(np.array(gestTensor),(1,0,2,3)))


class scutDataset(torch.utils.data.Dataset):

    """
    Dataset class for SCUT-DHGA
    """

    def __init__(self,
                 dataDir='./datasets/scut/scut/color_hand',
                 mode='train',
                 splitSize=0.2,
                 numFrames=64,
                 sample=0,
                 sampleMethod='sample',
                 sessionID=1):
        
        self.dataDir = dataDir # Path to the data directory
        self.mode = mode # mode: ['train','val','test']
        self.splitSize = splitSize # splitSize

        assert not(self.mode == 'val' and self.splitSize == 0.0), "No validation split"

        self.numFrames = numFrames # Total frames to be used
        self.numGestures = 6 # Total number of gestures
        self.sample = sample # Perform sampling is sample=1
        self.samplingMethod = sampleMethod # Sampling method, default: 'sample'
        self.sessionID = sessionID # Session ID for the test set

        self.folderList = [] # List to store path of data folders
        self.y_id = [] # List to store subject IDs
        self.y_gid = [] # List to store gesture IDs

        if(self.mode == 'train'):
            numSubjects = 143
            numInstances = 10
            for subject_id in range(numSubjects):    
                for gesture_id in range(self.numGestures):
                    for instance_id in range(int(numInstances-splitSize*numInstances)): # Train split
                        folderName = self.dataDir + '/1_1_'+str(subject_id)+'_'+str(gesture_id)+'_'+str(instance_id)+'/' # Name of the folder
                        self.folderList.append(folderName)
                        self.y_id.append(subject_id)
                        self.y_gid.append(gesture_id)  
                # print('++++++++++++++++++++++++++')
                # print('Processed Train Subject #: '+str(subject_id+1))

        if(self.mode == 'val'):
            numSubjects = 143
            numInstances = 10
            for subject_id in range(numSubjects):    
                for gesture_id in range(self.numGestures):
                    for instance_id in range(int(numInstances-splitSize*numInstances),numInstances): # Validation split
                        folderName = self.dataDir + '/1_1_'+str(subject_id)+'_'+str(gesture_id)+'_'+str(instance_id) # Name of the folder
                        self.folderList.append(folderName)
                        self.y_id.append(subject_id)
                        self.y_gid.append(gesture_id)
                # print('++++++++++++++++++++++++++')
                # print('Processed Validation Subject #: '+str(subject_id+1))

        if(self.mode == 'test'):
            numSubjects = 50
            numInstances = 10
            for subject_id in range(numSubjects):
                for gesture_id in range(self.numGestures):
                    # for session_id in range(1,1+numSessions):
                    for instance_id in range(numInstances):
                        folderName = self.dataDir + '/2_'+str(self.sessionID)+'_'+str(subject_id)+'_'+str(gesture_id)+'_'+str(instance_id) # Name of the folder
                        self.folderList.append(folderName)
                        self.y_id.append(subject_id)
                        self.y_gid.append(gesture_id)
                # print('++++++++++++++++++++++++++')
                # print('Processed Test Subject #: '+str(subject_id+1))


        if(self.mode != 'test'):
            self.folderList, self.y_id, self.y_gid = shuffle(self.folderList,
                                                            self.y_id,
                                                            self.y_gid,
                                                            random_state=42) # shuffling
    
    def __len__(self):
        return len(self.folderList)
    
    def __getitem__(self, idx):
        return {'data':processFolder(self.folderList[idx],
                                     numFrames=self.numFrames,
                                     sample=self.sample,
                                     sampleMethod=self.samplingMethod),
                'label':torch.tensor(self.y_id[idx]).type(torch.LongTensor)}
    
    def __getLabels__(self):
        return np.array(self.y_id), np.array(self.y_gid)
        
if __name__ == "__main__":

    # import matplotlib.pyplot as plt
    # img = processFrame('./datasets/scut/scut/color_hand/1_1_0_0_0/32.jpg')
    # print(img.shape)
    # print(np.max(img),np.min(img))
    # plt.imshow(np.transpose(img,(1,2,0)))
    # plt.show()

    # gestTensor = processFolder('./datasets/scut/scut/color_hand/1_1_0_0_0',numFrames=16,sampleMethod='sample')
    # print(gestTensor.shape)
    # print(gestTensor.numpy())
    # print(torch.max(gestTensor).item(),torch.min(gestTensor).item())

    dataset = scutDataset(dataDir='./datasets/scut/color_hand',
                          mode='test',
                          splitSize=0.2,
                          numFrames=64,
                          sample=0,
                          sampleMethod='sample',
                          sessionID=2
                          )
    print(dataset.__len__())

    device = torch.device('cuda:0')

    dataLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=8,
                                             shuffle=True,
                                             drop_last=False)
    
    for batch_idx, data in enumerate(dataLoader):
        print(data['data'].to(device).size(),
              data['label'].to(device).size())
        print(data['label'].detach().cpu().numpy())
        
    # for i in range(10):
    #     data = dataset.__getitem__(i)
    #     print(data['data'].shape, data['label'].item())
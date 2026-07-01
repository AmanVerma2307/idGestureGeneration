import torch
import torch.nn.functional as F
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


class motionModel(torch.nn.Module):
    
    """
    Model defined for pretrained motion maps
    """

    def __init__(self,
                 args,
                 I):
        
        super().__init__()
        self.normConv = torch.nn.Conv3d(args.motionModel_Cin,3,kernel_size=(3,3,3))
        self.model = getModel(args,
                              T=args.numFrames-1,
                              H=args.sizeH,
                              W=args.sizeW,
                              C=3,
                              I=I)
        
    def forward(self,x):
        x = F.pad(x,(1,1,1,1,1,1))
        x = self.normConv(x)
        x = F.relu(x)
        preds, embeddings = self.model(x)
        return preds, embeddings


    def predict(self,
                dataLoader,
                args):
        
        """
        Function to predict embeddings and outputs

        INPUTS:-
        1) dataLoader: The testSet loader with N samples
        2) args: Parsed arguments

        OUPUTS:-
        1) y_preds: Predicted labels of shape (N,)
        2) f_theta: Predicted embeddings of shape (N,d)
        """

        y_preds = []
        embeddings = []
        
        if(args.multi_gpu == 0):
            device = torch.device(args.device)

        for batch_idx, dataSample in enumerate(tqdm.tqdm(dataLoader,colour='yellow')):
            
            self.eval()
            with torch.set_grad_enabled(False):
                dense_id, f_theta = self.forward(dataSample['data'].to(device))

            for elemPreds, elemEmbeddings in zip(torch.argmax(dense_id,dim=-1).detach().cpu().numpy(),f_theta.detach().cpu().numpy()):
                y_preds.append(elemPreds)
                embeddings.append(elemEmbeddings)                

        return np.array(y_preds), np.array(embeddings)
    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelChoice',
                        type=str,
                        default="fstanet")
    parser.add_argument('--numFrames',
                        type=int,
                        default=64)
    parser.add_argument('--sizeH',
                        type=int,
                        default=128)
    parser.add_argument('--sizeW',
                        type=int,
                        default=128)
    
    args = parser.parse_args()
    device = torch.device('cuda:0')

    model = motionModel(args,
                        C=15,
                        I=143).to(device)
    input = torch.randn(size=(2,15,63,128,128)).to(device)
    op1, op2 = model(input)
    print(op1.size(),op2.size())

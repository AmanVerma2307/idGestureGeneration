import tqdm
import timm
import torch
import torchvision
import numpy as np

class resnet18Video(torch.nn.Module):

    """
    ResNet18 Video Class
    """

    def __init__(self,I):
        super().__init__()
        self.I = I
        self.backbone = timm.create_model('resnet18', pretrained=True) #torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        self.dense_id = torch.nn.Linear(in_features=512,out_features=self.I)

    def forward(self,x):
        n,c,t,h,w = x.size()
        embeddings = self.backbone(x.permute(0,2,1,3,4).contiguous().view(-1,c,h,w))
        embeddings = torch.mean(embeddings.view(-1,t,embeddings.shape[-1]),axis=1)
        op = self.dense_id(embeddings)
        return op, embeddings

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

            x = dataSample['data'].to(device)
            y_id = dataSample['label'].to(device)

            self.eval()
            with torch.set_grad_enabled(False):
                dense_id, f_theta = self.forward(x)
            
            for elemPreds, elemEmbeddings in zip(torch.argmax(dense_id,dim=-1).detach().cpu().numpy(),f_theta.detach().cpu().numpy()):
                y_preds.append(elemPreds)
                embeddings.append(elemEmbeddings)                

        return np.array(y_preds), np.array(embeddings)
    
if __name__ == "__main__":
    device = torch.device('cuda:0')
    input = torch.randn(size=(32,3,40,128,128)).to(device)
    model = resnet18Video(I=143).to(device)
    op1, op2 = model(input)
    print(op1.shape, op2.shape)

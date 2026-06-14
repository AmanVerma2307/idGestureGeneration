import torch
import torchvision

class resnet18Video(torch.nn.Module):

    """
    ResNet18 Video Class
    """

    def __init__(self,I):
        super().__init__()
        self.I = I
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        self.dense_id = torch.nn.Linear(in_features=512,out_features=self.I)

    def forward(self,x):
        n,c,t,h,w = x.size()
        embeddings = self.backbone(x.permute(0,2,1,3,4).contiguous().view(-1,c,h,w))
        embeddings = torch.mean(embeddings.view(-1,t,embeddings.shape[-1]),axis=1)
        op = self.dense_id(embeddings)
        return op, embeddings
    
if __name__ == "__main__":
    device = torch.device('cuda:0')
    input = torch.randn(size=(32,3,40,128,128)).to(device)
    model = resnet18Video(I=143).to(device)
    op1, op2 = model(input)
    print(op1.shape, op2.shape)

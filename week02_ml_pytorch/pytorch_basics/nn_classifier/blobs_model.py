import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobsModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
        
    def forward(self,x):
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        return x
model = BlobsModel().to(device)

        
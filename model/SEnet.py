import torch
from torch import nn

class senet(nn.Module):
    def __init__(self,chanel,ratio = 16):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(chanel,chanel//ratio,False),
            nn.ReLU(),
            nn.Linear(chanel//ratio,chanel,False),
            nn.Sigmoid(),

        )

    def forward(self,x):
            b,c,h,w = x.size()
            # b,c,h,w -> b,c,1,1
            ave = self.avg_pool(x).view([b,c])

            #b,c - >b ,c //ratio -> b,c -> b,c ,1 , 1
            fc = self.fc(ave).view([b,c,1,1])
            print(fc)
            return x* fc

model = senet(512)
print(model)
inputs = torch.ones([2,512,26, 26])

outputs = model(inputs)
print(outputs.shape)
import torch.nn as nn
import torch
import torch.nn.functional as F
from kornia.losses import SSIM
from kornia.filters import Laplacian,Sobel
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.lap=Laplacian(3)
        self.sob=Sobel()
        # kornia 里面的定义是 loss = (1 - SSIM) / 2

    def forward(self, out, gt):
        l1 = self.l1(out['image'], gt['gt'])
        lap_l1=self.l1(self.lap(out['image']),self.lap(gt['gt']))
        sob_l1=self.l1(self.sob(out['image']),self.sob(gt['gt']))
        tot = l1+0.5*lap_l1+0.2*sob_l1
        # return {'tot': tot, 'L1': l1}
        return {'loss': tot, 'L1': l1, 'lap_l1': lap_l1,'sob_l1':sob_l1}


def get():
    return MyLoss()
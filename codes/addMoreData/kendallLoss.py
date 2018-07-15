#encoding:utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class kendallloss(nn.Module):
    def __init__(self):
        super(kendallloss,self).__init__()
        
    def forward(self,predict,target):
        num = 0
        for i in range(1, len(target)):
            for j in range(i):
                #num += torch.sign(predict[i][0] - predict[j][0]) * torch.sign(target[i] - target[j])
                num += F.hardtanh(predict[i] - predict[j]) * F.hardtanh(target[i] - target[j])
                
        x =  1 - 2 * num / (len(target) * (len(target) - 1))
        #print(torch.autograd.grad(x, predict))
        return x
    




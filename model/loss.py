import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, input, target):
        loss = torch.mean((input - target)**2)
        return loss

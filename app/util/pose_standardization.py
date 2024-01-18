import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import math
import os

def pose_standardization(pose):
    frameLength = pose.shape[0]
    pose = pose.reshape((frameLength, 25, 3, 1))
    # print(kkk[0])
    zzz = np.zeros((frameLength, 25, 3, 1))

    pose = np.concatenate((pose, zzz), axis=3)
    pose = pose.transpose((2, 0, 1, 3))
    print('pose标准化',pose.shape)
    return pose
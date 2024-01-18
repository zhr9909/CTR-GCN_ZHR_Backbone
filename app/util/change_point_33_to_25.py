import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import math
import os

def change_point_33_to_25(pose):


    all_frame_keypoints = np.empty((0, 25, 3), dtype=float)
    for i in range(pose.shape[0]):
        keypoints = pose[i]
        one_frame_keypoints = np.empty((0, 3), dtype=float)

        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([(keypoints[23][0]+keypoints[24][0])/2,(keypoints[23][1]+keypoints[24][1])/2,(keypoints[23][2]+keypoints[24][2])/2]))) #1
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([(keypoints[23][0]+keypoints[24][0]+keypoints[11][0]+keypoints[12][0])/4,(keypoints[23][1]+keypoints[24][1]+keypoints[11][1]+keypoints[12][1])/4,(keypoints[23][2]+keypoints[24][2]+keypoints[11][2]+keypoints[12][2])/4]))) #2
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([((keypoints[11][0]+keypoints[12][0])/2+keypoints[0][0])/2,((keypoints[11][1]+keypoints[12][1])/2+keypoints[0][1])/2,((keypoints[11][2]+keypoints[12][2])/2+keypoints[0][2])/2]))) #3
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[0][0],keypoints[0][1],keypoints[0][2]]))) #4
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[12][0],keypoints[12][1],keypoints[12][2]]))) #5
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[14][0],keypoints[14][1],keypoints[14][2]]))) #6
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[16][0],keypoints[16][1],keypoints[16][2]]))) #7
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[20][0],keypoints[20][1],keypoints[20][2]]))) #8
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[11][0],keypoints[11][1],keypoints[11][2]]))) #9
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[13][0],keypoints[13][1],keypoints[13][2]]))) #10
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[15][0],keypoints[15][1],keypoints[15][2]]))) #11
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[19][0],keypoints[19][1],keypoints[19][2]]))) #12
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[24][0],keypoints[24][1],keypoints[24][2]]))) #13
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[26][0],keypoints[26][1],keypoints[26][2]]))) #14
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[28][0],keypoints[28][1],keypoints[28][2]]))) #15
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[32][0],keypoints[32][1],keypoints[32][2]]))) #16
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[23][0],keypoints[23][1],keypoints[23][2]]))) #17
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[25][0],keypoints[25][1],keypoints[25][2]]))) #18
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[27][0],keypoints[27][1],keypoints[27][2]]))) #19
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[31][0],keypoints[31][1],keypoints[31][2]]))) #20
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([(keypoints[11][0]+keypoints[12][0])/2,(keypoints[11][1]+keypoints[12][1])/2,(keypoints[11][2]+keypoints[12][2])/2]))) #21
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[18][0],keypoints[18][1],keypoints[18][2]]))) #22
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[22][0],keypoints[22][1],keypoints[22][2]]))) #23
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[17][0],keypoints[17][1],keypoints[17][2]]))) #24
        one_frame_keypoints = np.vstack((one_frame_keypoints, np.array([keypoints[21][0],keypoints[21][1],keypoints[21][2]]))) #25

        all_frame_keypoints = np.vstack([all_frame_keypoints, one_frame_keypoints[None, ...]])

    print('all_frame_keypoints.shape: ',all_frame_keypoints.shape)
    return all_frame_keypoints
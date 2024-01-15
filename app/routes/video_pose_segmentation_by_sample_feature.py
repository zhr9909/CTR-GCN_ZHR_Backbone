from flask import Flask, request, Response
import yaml
import numpy as np
from app.routes.push_pose_to_GCN import push_pose_to_GCN_function
import matplotlib.pyplot as plt
from YOLO.predict_one_image import predict_one_image



def video_pose_segmentation_by_sample_feature_function():
    data = request.json
    try:
        pose = data['pose']
        actionType = data['actionType']
    except KeyError as e:
        print('请求的JSON参数不全')
        return '请求的JSON参数不全'

    print(len(pose)//3//25//2)
    pose = np.array(pose).reshape((3,len(pose)//3//25//2,25,2))
    videoFeature = push_pose_to_GCN_function(pose)

    try:
        sampleFeature = np.load('app/sampleVideo/{}.npy'.format(actionType))
    except FileNotFoundError as e:
        return '动作类型未找到'
    
    # 接下来进行待检测视频和标准样例间的特征相似度矩阵构建
    normsSampleFeature = np.linalg.norm(sampleFeature, axis=1, keepdims=True)
    normalized_sampleFeature = sampleFeature / normsSampleFeature #标准样例的归一化特征

    normsVideoFeature = np.linalg.norm(videoFeature, axis=1, keepdims=True)
    normalized_videoFeature = videoFeature / normsVideoFeature #待检测视频的归一化特征

    similirityMatrix = np.dot(normalized_sampleFeature, normalized_videoFeature.T) #!!!特征形似度矩阵!!!


    for i in range(similirityMatrix.shape[1]//64):
        gululu = similirityMatrix[32:64,i*64:(i+1)*64]

        fig = plt.figure(figsize=(150, 10),dpi=12)
        plt.imshow(gululu, cmap='coolwarm', interpolation='nearest',vmin=0,vmax=0.7)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('app的目标检测尝试.png',bbox_inches='tight', pad_inches=0)


        canvas = fig.canvas
        canvas.draw()
        fig_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        fig_array = fig_array.reshape(canvas.get_width_height()[::-1] + (3,))

        # 找到非白像素的索引
        non_white_pixels = np.where(fig_array < 255)

        # 获取边界框
        min_y, max_y = np.min(non_white_pixels[0]), np.max(non_white_pixels[0])
        min_x, max_x = np.min(non_white_pixels[1]), np.max(non_white_pixels[1])

        # 裁剪图像
        fig_array = fig_array[min_y:max_y+1, min_x:max_x+1, :]

        fig_array = np.ascontiguousarray(fig_array[:, :, [2, 1, 0]])
        print(fig_array.shape)
    
        predict_one_image(fig_array)



    return '标准样例提取并保存成功'


from flask import Flask, request, Response, jsonify
import yaml
import numpy as np
import json
from app.routes.push_pose_to_GCN import push_pose_to_GCN_function
from app.util.change_point_33_to_25 import change_point_33_to_25
from app.util.pose_standardization import pose_standardization
from app.util.houchuli import houchuli
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
    
    print(type(pose))
    pose = np.array(json.loads(pose))
    
    # 将不足64倍数的帧进行补齐
    if int(pose.shape[0]) % 64 != 0:
        selected_array = pose[int(pose.shape[0])-1]
        copied_arrays = np.repeat(selected_array[np.newaxis, :], (int(pose.shape[0])//64+1)*64-int(pose.shape[0]), axis=0)
        pose = np.concatenate((pose, copied_arrays), axis=0)

    print('pose.shape: ',pose.shape)
    # np.save('app/sampleVideo/深蹲/深蹲视频骨骼序列3.npy',pose)
    pose = change_point_33_to_25(pose)
    print('转换后的pose.shape',pose.shape)
    jsonPose = pose.tolist()
    pose = pose_standardization(pose)
    # np.save('app/sampleVideo/深蹲/stu8_69.npy',pose)

    videoFeature = push_pose_to_GCN_function(pose)
    # np.save('app/sampleVideo/深蹲/stu3_65特征序列.npy',videoFeature[200:220,:])

    try:
        sampleFeature = np.load('app/sampleVideo/{}特征序列.npy'.format(actionType))
    except FileNotFoundError as e:
        return '动作类型未找到'
    
    # 接下来进行待检测视频和标准样例间的特征相似度矩阵构建
    normsSampleFeature = np.linalg.norm(sampleFeature, axis=1, keepdims=True)
    normalized_sampleFeature = sampleFeature / normsSampleFeature #标准样例的归一化特征
    # normalized_sampleFeature =np.concatenate((normalized_sampleFeature[:,0:1280],normalized_sampleFeature[:,2048:2304],normalized_sampleFeature[:,3072:5376]), axis=1) 

    normsVideoFeature = np.linalg.norm(videoFeature, axis=1, keepdims=True)
    normalized_videoFeature = videoFeature / normsVideoFeature #待检测视频的归一化特征
    # normalized_videoFeature =np.concatenate((normalized_videoFeature[:,0:1280],normalized_videoFeature[:,2048:2304],normalized_videoFeature[:,3072:5376]), axis=1) 

    similirityMatrix = np.dot(normalized_sampleFeature, normalized_videoFeature.T) #!!!特征形似度矩阵!!!

    print('特征形似度矩阵的形状：',similirityMatrix.shape)

    YOLO_Predict_result_list = []
    for i in range(similirityMatrix.shape[1]//64):
        gululu = similirityMatrix[:,i*64:(i+1)*64]

        fig = plt.figure(figsize=(150, 10),dpi=12)
        plt.imshow(gululu, cmap='coolwarm', interpolation='nearest',vmin=0,vmax=0.55)
        plt.xticks([])
        plt.yticks([])
        # plt.savefig('app的目标检测尝试.png',bbox_inches='tight', pad_inches=0)


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

        subResult = predict_one_image(fig_array) # 每次YOLO推理得到的结果(conf,x,y,w,h)
        if len(subResult) > 0:
            for k in range(len(subResult)):
                # 处理得到真实的推理结果时间下标
                subResult[k] = [subResult[k][0],subResult[k][1]*64+i*64,subResult[k][2],subResult[k][3]*64,subResult[k][4]] 

        YOLO_Predict_result_list.append(subResult)

    for i in range(len(YOLO_Predict_result_list)):
        print('YOLO_Predict_result_list:',YOLO_Predict_result_list[i])
    
    Final_YOLO_Predict_result_list = []
    for i in range(len(YOLO_Predict_result_list)):
        for j in range(len(YOLO_Predict_result_list[i])):
            x1 = YOLO_Predict_result_list[i][j][1] - YOLO_Predict_result_list[i][j][3]/2
            x2 = YOLO_Predict_result_list[i][j][1] + YOLO_Predict_result_list[i][j][3]/2
            y1 = YOLO_Predict_result_list[i][j][2] - YOLO_Predict_result_list[i][j][4]/2
            y2 = YOLO_Predict_result_list[i][j][2] + YOLO_Predict_result_list[i][j][4]/2
            Final_YOLO_Predict_result_list.append([YOLO_Predict_result_list[i][j][0],int(x1),int(x2),y1,y2])

    Final_YOLO_Predict_result_list = sorted(Final_YOLO_Predict_result_list, key=lambda x: x[1])
    print('Final_YOLO_Predict_result_list',Final_YOLO_Predict_result_list)
    Ending = houchuli(Final_YOLO_Predict_result_list)
    for i in range(len(Ending)):
        print('Ending({}): '.format(i),Ending[i])

    jsonBody = {}
    jsonBody['message'] = '标准样例提取并保存成功'
    Ending.append([0,448])
    jsonBody['data'] = Ending
    jsonBody['pose'] = jsonPose
    
    return jsonBody



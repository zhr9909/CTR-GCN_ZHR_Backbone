from flask import Flask, request, Response
import yaml
import numpy as np
from app.routes.push_pose_to_GCN import push_pose_to_GCN_function



def push_sample_video_pose_to_GCN_and_save_function():
    data = request.json
    try:
        pose = data['pose']
        name = data['name']
    except KeyError as e:
        print('请求的JSON参数不全')
        return '请求的JSON参数不全'

    pose = np.array(pose).reshape((3,len(pose)//3//25//2,25,2))
    result = push_pose_to_GCN_function(pose)
    print(result.shape)

    
    np.save('app/sampleVideo/{}.npy'.format(name), result)

    return '标准样例提取并保存成功'


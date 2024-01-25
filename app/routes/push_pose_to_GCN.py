from flask import Flask, request, Response
from app.routes.CTR_ZHR.main_zhr_predict_act_for_app import get_parser,Processor,init_seed
import yaml
import numpy as np

parser = get_parser()

# load arg form config file
p = parser.parse_args()

p.config = 'app/routes/CTR_ZHR/config.yaml'


if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.set_defaults(**default_arg)

arg = parser.parse_args()
arg.device = 0
print(type(arg))
init_seed(arg.seed)
print(arg)
processor = Processor(arg)


def push_pose_to_GCN_function(pose):
    result = processor.train(pose)
    # print(type(request.files))
    # data = request.json
    # print('接收到的数据: ',data)
    # print(data['pose'][2])
    # if 'file' not in request.files:
    #     return 'No file part'

    # file = request.files['file']
    # if file.filename == '':
    #     return 'No selected file'

    # 这里假设上传的文件是视频文件，并进行相应处理
    # 你可以将这个视频流保存到服务器，或者对视频进行处理等操作
    # file.save('uploaded_video.mp4')  # 保存上传的视频文件到服务器


    return result



from PIL import Image
from ultralytics import YOLO
import numpy as np

def predict_one_image(fig_array):
    # Load a pretrained YOLOv8n model
    model = YOLO('runs/detect/train24/weights/best.pt')

    # 填写预测图片的路径'，
    results = model(fig_array, stream=True, save=True)  # results list

    # 如果后续需要目标的位置，置信度等信息，可以从results里面寻找
    output_line = []
    resultList = []
    for r in results:
        conf = r.boxes.conf.cpu().detach().numpy()
        xywhn = r.boxes.xywhn.cpu().detach().numpy()
        result_label = np.concatenate((conf[:, None], xywhn), axis=1)
        # xywhn = xywhn.append(conf)
        for i in range(len(conf)):
            # output_line = output_line + result_label[i].tolist()
            print(conf)
            print(xywhn)
            print(result_label[i].tolist())
            resultList.append(result_label[i].tolist())
    
    return resultList


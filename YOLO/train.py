from ultralytics import YOLO


# 自己训练的时候可以自己选择yolov8的版本，直接修改配置文件后面的后缀即可,data路径填写绝对路径
if __name__ == '__main__':
    model = YOLO('yolov8s.yaml').load('weight/yolov8s.pt')
    model.train(data='/home/zhanghaoran/zhr_downloads/CTR-GCN_ZHR_Backbone/YOLO/datasets/data.yaml', epochs=299,
                imgsz=640, patience=50, batch=2, save=True,
                optimizer='auto', val=True,)
#现在的步骤不用理会，你的电脑上面不会出现这个报错
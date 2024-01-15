
from PIL import Image
from ultralytics import YOLO

def predict_one_image(fig_array):
    # Load a pretrained YOLOv8n model
    model = YOLO('runs/detect/train21/weights/epoch60.pt')

    # 填写预测图片的路径'，
    results = model(fig_array, stream=True,save=True)  # results list

    # 如果后续需要目标的位置，置信度等信息，可以从results里面寻找
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        # im.save('results.jpg')  # save image


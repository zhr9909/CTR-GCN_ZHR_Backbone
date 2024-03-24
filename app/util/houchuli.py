import numpy as np

def houchuli(Final_YOLO_Predict_result_list):
    Ending = []
    for i in range(len(Final_YOLO_Predict_result_list)):
        subResult = Final_YOLO_Predict_result_list[i]
    # for subResult in Final_YOLO_Predict_result_list:
        if subResult[0] < 0.5: #可信度太低
            continue
        if len(Ending) == 0: #加入第一个
            Ending.append(subResult[1:])
            # if Ending[-1][1]%64
            continue
        elif subResult[1]-3 >= Ending[-1][1] and subResult[1]-8 <= Ending[-1][1] and subResult[3]+0.2 >= Ending[-1][3]: #
            Ending[-1][1] = subResult[2]
            Ending[-1][3] = subResult[4]
            continue
        elif subResult[1]-3 >= Ending[-1][1]:
            Ending.append(subResult[1:])
            continue
        elif subResult[1]-3 <= Ending[-1][1]:
            if subResult[1] > Ending[-1][1] and subResult[4] - subResult[3]>0.6 and subResult[3]+0.2 < Ending[-1][3]:
                Ending.append(subResult[1:])
            else:
                Ending[-1][1] = max(subResult[2],Ending[-1][1])
                Ending[-1][2] = min(subResult[3],Ending[-1][2])
                Ending[-1][3] = max(subResult[4],Ending[-1][3])
            continue
    
    Ending = [sub for sub in Ending if sub[1]-sub[0]>8 and sub[3] - sub[2]>0.7]
    for sub in Ending:
        xlength = sub[1] - sub[0]
        dx_left = sub[2] * xlength
        dx_right = (1-sub[3]) * xlength
        sub[0] -= int(dx_left)
        if sub[0] < 0:
            sub[0] = 0
        sub[1] += int(dx_right)
    return [sub[0:2] for sub in Ending]


# 让起始帧的身体居中，后续的坐标都按照起始坐标来移动
def bodyCentor(pose):
    dx = 0 - pose[0, 0, 20, 0]
    dy = 0 - pose[1, 0, 20, 0]
    for i in range(0, pose.shape[1]):
        for j in range(25):
            pose[0, i, j, 0] += dx
            pose[1, i, j, 0] += dy
    return pose

def changeSkeletonLength(point):
    # 肩膀宽度
    lengthShoulder = 0
    for i in range(0, point.shape[1]):
        lengthShoulder += np.linalg.norm(
            np.array([point[0, i, 4, 0] - point[0, i, 8, 0], point[1, i, 4, 0] - point[1, i, 8, 0]]))
    lengthShoulder = lengthShoulder / point.shape[1]
    # 脊椎长度
    lengthSpine = 0
    for i in range(0, point.shape[1]):
        lengthSpine += np.linalg.norm(
            np.array([point[0, i, 0, 0] - point[0, i, 20, 0], point[1, i, 0, 0] - point[1, i, 20, 0]]))
    lengthSpine = lengthSpine / point.shape[1]

    print('肩膀宽度', lengthShoulder)
    print('脊椎长度', lengthSpine)

    if lengthShoulder > 0.12 and lengthSpine > 0.12:
        proportion = 1 / min(lengthShoulder / 0.12, lengthSpine / 0.12)
    elif lengthShoulder > 0.12 and lengthSpine < 0.12:
        proportion = 1 / min(lengthShoulder / 0.12, 0.12 / lengthSpine)
    elif lengthShoulder < 0.12 and lengthSpine > 0.12:
        proportion = 1 / min(0.12 / lengthShoulder, lengthSpine / 0.12)
    elif lengthShoulder < 0.12 and lengthSpine < 0.12:
        proportion = 1 / min(0.12 / lengthShoulder, 0.12 / lengthSpine)
    for i in range(0, point.shape[1]):
        for j in range(25):
            x = point[0, i, j, 0]
            y = point[1, i, j, 0]
            point[0, i, j, 0] = x * proportion
            point[1, i, j, 0] = y * proportion
    return point


def houchuli(Final_YOLO_Predict_result_list):
    Ending = []
    for subResult in Final_YOLO_Predict_result_list:
        if subResult[0] < 0.3: #可信度太低
            continue
        if len(Ending) == 0: #加入第一个
            Ending.append(subResult[1:])
            continue
        if subResult[1]-3 >= Ending[-1][1] and subResult[3]+0.2 >= Ending[-1][3]: #
            Ending[-1][1] = subResult[2]
            Ending[-1][3] = subResult[4]
            continue
        if subResult[1]-3 >= Ending[-1][1] and subResult[3]+0.2 <= Ending[-1][3]:
            Ending.append(subResult[1:])
            continue
        if subResult[1]-3 <= Ending[-1][1]:
            Ending[-1][1] = max(subResult[2],Ending[-1][1])
            Ending[-1][3] = max(subResult[4],Ending[-1][3])
            continue
    
    return [sub[0:2] for sub in Ending]
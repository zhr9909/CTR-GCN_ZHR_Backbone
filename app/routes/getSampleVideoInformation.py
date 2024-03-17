from flask import Flask, request, Response, jsonify
import os
import numpy as np

def getSampleVideoInformation_function():
    fileDirection = 'app/sampleVideo/'
    fileList = os.listdir(fileDirection)
    fileInformationList = []
    for file in fileList:
        file_path = os.path.join(fileDirection, file)
        if os.path.isfile(file_path):
            print(file)
            fileInformation = []
            fileInformation.append(file[:-8])
            fileInformation.append(np.load('app/sampleVideo/{}'.format(file)).shape[0])
            fileInformationList.append(fileInformation)
    jsonBody = {}
    jsonBody['data'] = fileInformationList
    return jsonBody
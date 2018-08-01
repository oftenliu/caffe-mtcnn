#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import sys
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./"))
sys.path.insert(0, rootPath)
from model.mtcnnmodel import mtcnn_pnet, mtcnn_rnet,mtcnn_onet

from util.loader import TestLoader
from detect.ronet_detect import ROnetDetect
from detect.pnet_detect import PnetDetect
from detect.detect import MtcnnDetector

import cv2
import argparse

def test(stage, testFolder):
    print("Start testing in %s"%(testFolder))
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/pnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = PnetDetect(mtcnn_pnet,modelPath) 
    if stage in ['rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/rnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = ROnetDetect(mtcnn_rnet, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/onet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = ROnetDetect(mtcnn_onet, 48, 1, modelPath)
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])

    testImages = []
    for name in os.listdir(testFolder):
        testImages.append(os.path.join(testFolder, name))
    testDatas = TestLoader(testImages)
    # Now to detect
    allBoxes, allLandmarks = mtcnnDetector.detect_face(testDatas)
    print("\n")
    # Save it
    for idx, imagePath in enumerate(testImages):
        image = cv2.imread(imagePath)
        for bbox in allBoxes[idx]:
            cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        allLandmark = allLandmarks[idx]
        if allLandmark is not None: # pnet and rnet will be ignore landmark
            for landmark in allLandmark:
                for i in range(int(len(landmark)/2)):
                    cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
        savePath = os.path.join(rootPath, 'testing', 'results_%s'%(stage))
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        cv2.imwrite(os.path.join(savePath, "result_%d.jpg" %(idx)), image)
        print("Save image to %s"%(savePath))

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='onet', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    stage = 'onet'
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # Support stage: pnet, rnet, onet
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # set GPU
    test(stage, os.path.join(rootPath, "images"))


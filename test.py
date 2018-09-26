#coding:utf-8
import numpy as np
import os
import sys
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./"))
sys.path.insert(0, rootPath)


from mtcnn_config import config
from util.loader import TestLoader
from detect.detect import MtcnnDetector

import cv2
import argparse

def test(testFolder,stage):
    print("Start testing in %s"%(testFolder))
    if stage in ["pnet"]:
        net = ['caffe-pnet/pnet.prototxt', 'testmodel/p.caffemodel']
        #net = ['pmodel/p.prototxt', 'pmodel/p.caffemodel']
    if stage in ["rnet"]:
        net = ['caffe-pnet/pnet.prototxt', 'testmodel/p.caffemodel', 'caffe-rnet/rnet.prototxt', 'testmodel/r.caffemodel']
    if stage in ["onet"]:    
        net = ['caffe-pnet/pnet.prototxt', 'testmodel/p.caffemodel', 'caffe-rnet/rnet.prototxt', 'testmodel/r.caffemodel', 'caffe-onet/onet.prototxt', 'testmodel/o.caffemodel']

    testImages = []
    for name in os.listdir(testFolder):
        testImages.append(os.path.join(testFolder, name))

    mtcnn_detector = MtcnnDetector(net,min_face_size=120,stride=2,threshold=[0.6, 0.7, 0.7])
    test_data = TestLoader(testImages)
    # do detect
    detections, allLandmarks = mtcnn_detector.detect_face(test_data)
    # save detect result
    print("\n")
    # Save it
    for idx, imagePath in enumerate(testImages):
        image = cv2.imread(imagePath)
        for bbox in detections[idx]:
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
    stage = args.stage
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # Support stage: pnet, rnet, onet
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # set GPU
    test(os.path.join(rootPath, "images"),stage)


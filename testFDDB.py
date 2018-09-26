#coding:utf-8
import sys
sys.path.append("..")
import argparse
from mtcnn_config import config
from util.loader import TestLoader
from detect.detect import MtcnnDetector
import cv2
import os
data_dir = './fddb'
out_dir = './fddb/Res'

def get_imdb_fddb(data_dir):
    imdb = []
    nfold = 10
    for n in range(nfold):
        file_name = 'FDDB-folds/FDDB-fold-%02d.txt' % (n + 1)
        file_name = os.path.join(data_dir, file_name)
        fid = open(file_name, 'r')
        image_names = []
        for im_name in fid.readlines():
            image_names.append(im_name.strip('\n'))      
        imdb.append(image_names)
    return imdb        



if __name__ == "__main__":
    stage = "onet"
    if stage in ["pnet"]:
        net = ['caffe-pnet/pnet.prototxt', 'testmodel/p.caffemodel']
        #net = ['pmodel/p.prototxt', 'pmodel/p.caffemodel']
    if stage in ["rnet"]:
        net = ['caffe-pnet/pnet.prototxt', 'testmodel/p.caffemodel', 'caffe-rnet/rnet.prototxt', 'testmodel/r.caffemodel']
    if stage in ["onet"]:    
        net = ['caffe-pnet/pnet.prototxt', 'testmodel/p.caffemodel', 'caffe-rnet/rnet.prototxt', 'testmodel/r.caffemodel', 'caffe-onet/onet.prototxt', 'testmodel/o.caffemodel']

    mtcnn_detector = MtcnnDetector(net,min_face_size=120,stride=2,threshold=[0.4, 0.6, 0.7])
    
    
    
    imdb = get_imdb_fddb(data_dir)
    nfold = len(imdb)    
    for i in range(nfold):
        image_names = imdb[i]
        print(image_names)
        dets_file_name = os.path.join(out_dir, 'fold-%02d-out.txt' % (i + 1))
        fid = open(dets_file_name,'w')
        sys.stdout.write('%s ' % (i + 1))
        image_names_abs = [os.path.join(data_dir,'originalPics',image_name+'.jpg') for image_name in image_names]
        test_data = TestLoader(image_names_abs)
        all_boxes, allLandmarks = mtcnn_detector.detect_face(test_data)
       
        for idx,im_name in enumerate(image_names):
            img_path = os.path.join(data_dir,'originalPics',im_name+'.jpg')
            image = cv2.imread(img_path)
            boxes = all_boxes[idx]
            if boxes is None:
                fid.write(im_name+'\n')
                fid.write(str(1) + '\n')
                fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                continue
            fid.write(im_name+'\n')
            fid.write(str(len(boxes)) + '\n')
            
            for box in boxes:
                fid.write('%f %f %f %f %f\n' % (float(box[0]), float(box[1]), float(box[2]-box[0]+1), float(box[3]-box[1]+1),box[4]))                
                       
        fid.close()
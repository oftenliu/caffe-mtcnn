#!/usr/bin/env python2.7
# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long
import sys
import os
sys.path.append('/home/ulsee/often/caffe/python')
import shutil
import random
import argparse
import cv2
import lmdb
import caffe
import numpy as np
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from util.log import get_logger


logger = get_logger()

G8 = 8*1024*1024*1024
G16 = 2*G8
G24 = 3*G8
G32 = 4*G8

def remove_if_exists(dbfile):
    if os.path.exists(dbfile):
      logger.info('remove %s'%dbfile)
      shutil.rmtree(dbfile)

    

def iter_all_data(net, iterType,num):
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    if net not in ['pnet', 'rnet', 'onet']:
        raise Exception("The net type error!")
    if not os.path.isfile(os.path.join(saveFolder, 'pos.txt')):
        raise Exception("Please gen pos.txt in first!")
    if not os.path.isfile(os.path.join(saveFolder, 'landmark.txt')):
        raise Exception("Please gen landmark.txt in first!") 

    Data_Ratio = {'neg':3, 'pos':1,'part':1,'landmark':2,}
    if iterType in ['pos', 'neg', 'part', 'landmark']:
        base_num = num
        with open(os.path.join(saveFolder, '%s.txt'%(iterType))) as f:
            data = f.readlines()

        if len(data) > base_num * Data_Ratio[iterType]:
            data_keep = np.random.choice(len(data), size=base_num * Data_Ratio[iterType], replace=False)
        else:
            data_keep = np.random.choice(len(data), size=len(data), replace=False)            
        print('The [%s] number = [%d].\n'%(iterType,data_keep.size))
        for i in data_keep:
            yield data[i]

    else:
        return





def process_image_withoutcoder(filename):
    image = cv2.imread(filename)
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    #python读取的图片文件格式为H×W×K，需转化为K×H×W  batch_size*channel*H*W      
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = (image - 127.5)/127.5
    height = image.shape[2]
    width = image.shape[1]

    image = image.tostring()
    return image, height, width



def parse_data(line):
    info = line.strip().split(' ')
    data_example = dict()
    bbox = dict()
    data_example['filename'] = info[0]
    data_example['label'] = int(info[1])
    bbox['xmin'] = 0
    bbox['ymin'] = 0
    bbox['xmax'] = 0
    bbox['ymax'] = 0
    bbox['xlefteye'] = 0
    bbox['ylefteye'] = 0
    bbox['xrighteye'] = 0
    bbox['yrighteye'] = 0
    bbox['xnose'] = 0
    bbox['ynose'] = 0
    bbox['xleftmouth'] = 0
    bbox['yleftmouth'] = 0
    bbox['xrightmouth'] = 0
    bbox['yrightmouth'] = 0
    if len(info) == 6:
        bbox['xmin'] = float(info[2])
        bbox['ymin'] = float(info[3])
        bbox['xmax'] = float(info[4])
        bbox['ymax'] = float(info[5])
    if len(info) == 12:
        bbox['xlefteye'] = float(info[2])
        bbox['ylefteye'] = float(info[3])
        bbox['xrighteye'] = float(info[4])
        bbox['yrighteye'] = float(info[5])
        bbox['xnose'] = float(info[6])
        bbox['ynose'] = float(info[7])
        bbox['xleftmouth'] = float(info[8])
        bbox['yleftmouth'] = float(info[9])
        bbox['xrightmouth'] = float(info[10])
        bbox['yrightmouth'] = float(info[11])
    data_example['bbox'] = bbox
    return data_example

def put_db(txn, dataset):
    '''
      #data format:[i,image_data,class_label,roi,landmark]
    '''
    for data in dataset:
        data_key = '%08d_data'%data[0]
        txn.put(data_key.encode(), data[1])

        label_key = '%08d_label'%data[0]
        txn.put(label_key.encode(), data[2])


        bbox_key = '%08d_bbox'%data[0]
        txn.put(bbox_key.encode(), data[3])

        landmark_key = '%08d_landmark'%data[0]
        txn.put(landmark_key.encode(), data[4]) 

#txn.put(str_id.encode('ascii'), datum.SerializeToString())
def write_db(dbname,net, iterType,num,shuffling):
    dataset = []
    i = 0
    db = lmdb.open(dbname, map_size=G24)
    txn = db.begin(write=True)
    for line in iter_all_data(net, iterType,num):
        data_example = parse_data(line)
        image_data, height, width = process_image_withoutcoder(data_example['filename'])
        class_label = data_example['label']
        bbox = data_example['bbox']

        roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
        landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                   bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]

        class_label = np.asarray(class_label, dtype=np.int32)
        #print(class_label)
        class_label = class_label.tostring()
        #print(class_label)
        roi = np.asarray(roi, dtype=np.float32)
        roi = roi.tostring()  # float32
        landmark = np.asarray(landmark, dtype=float)
        landmark = landmark.astype(np.float32).tostring()  # float32

        dataset.append([i,image_data,class_label,roi,landmark])
    
        
        i = i + 1
        if i % 1000 == 0:
            if shuffling:
                np.random.shuffle(dataset)
            put_db(txn,dataset)
            txn.commit()
            txn = db.begin(write = True)
            dataset = []
        if i % 10000 == 0:
            logger.info('Processed [%d] files.'%i)
    if i % 1000 != 0:
        if shuffling:
            np.random.shuffle(dataset)
        put_db(txn,dataset)
        logger.info('Processed [%d] files.'%i)
        print('Processed [%d] files.'%i)
    txn.put(('size').encode(), str(i).encode())
    txn.commit()
    db.close()
    logger.info('Create [ %s ] [ num %d] Finish'%(dbname,i))
    print('Create [ %s ] Finish'%dbname)

def gen_lmdb(filename, net, type,num,shuffling = True):
    remove_if_exists(filename)

    sizeOfNet = {"pnet": 12, "rnet": 24, "onet": 48}
    if net not in sizeOfNet:
        raise Exception("The net type error!")

    write_db(filename,net,type,num,shuffling)


def start(net):

    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    with open(os.path.join(saveFolder, 'pos.txt'), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(saveFolder, 'neg.txt'), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(saveFolder, 'part.txt'), 'r') as f:
        part = f.readlines()
    with open(os.path.join(saveFolder, 'landmark.txt'), 'r') as f:
        landmark = f.readlines()
    # keep sample ratio [neg, pos, part] = [3, 1, 1, 2]
    base_num = min([len(neg), len(pos), len(part), len(landmark)])


    for n in ['part', 'neg','pos', 'landmark']:
        filename = os.path.join(saveFolder, "%sdb"%(n))
        gen_lmdb(filename, net, n,base_num)
    # Finally, write the labels file:
    logger.info('\nFinished converting the MTCNN dataset!')







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                          default='unknow', type=str)

    args = parser.parse_args()

    stage = args.stage
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")

    start(stage)


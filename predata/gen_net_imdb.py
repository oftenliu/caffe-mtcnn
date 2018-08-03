import numpy as np
import numpy.random as npr
import  pickle
import sys
import cv2
import os
import argparse

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)



def __iter_all_data(net, iterType):
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    if net not in ['pnet', 'rnet', 'onet']:
        raise Exception("The net type error!")
    if not os.path.isfile(os.path.join(saveFolder, 'pos.txt')):
        raise Exception("Please gen pos.txt in first!")
    if not os.path.isfile(os.path.join(saveFolder, 'landmark.txt')):
        raise Exception("Please gen landmark.txt in first!")
    if iterType == 'all':
        with open(os.path.join(saveFolder, 'pos.txt'), 'r') as f:
            pos = f.readlines()
        with open(os.path.join(saveFolder, 'neg.txt'), 'r') as f:
            neg = f.readlines()
        with open(os.path.join(saveFolder, 'part.txt'), 'r') as f:
            part = f.readlines()
        # keep sample ratio [neg, pos, part] = [3, 1, 1]
        base_num = min([len(neg), len(pos), len(part)])
        base_num = 10
        if len(neg) > base_num * 3:
            neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=False)
        else:
            neg_keep = np.random.choice(len(neg), size=len(neg), replace=False)
        pos_keep = np.random.choice(len(pos), size=base_num, replace=False)
        part_keep = np.random.choice(len(part), size=base_num, replace=False)


        print("neg_keep = [%d]"%(len(neg_keep)))
        for i in pos_keep:
            yield pos[i]
        for i in neg_keep:
            yield neg[i]
        for i in part_keep:
            yield part[i]
        #for item in open(os.path.join(saveFolder, 'landmark.txt'), 'r'):
         #   yield item
    elif iterType in ['pos', 'neg', 'part', 'landmark']:
        for line in open(os.path.join(saveFolder, '%s.txt'%(iterType))):
            yield line
    else:
        raise Exception("Unsupport iter type.")



def process_image_withoutcoder(filename):
    image = cv2.imread(filename)
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    image = np.swapaxes(image, 0, 2) #python读取的图片文件格式为H×W×K，需转化为K×H×W  batch_size*channel*H*W  
    image = (image - 127.5)/127.5

    height = image.shape[2]
    width = image.shape[1]

    return image, height, width



def __get_dataset(net, iterType):
    dataset = []
    i = 0
    for line in __iter_all_data(net, iterType):
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

        filename = data_example['filename']
        image_data, height, width = process_image_withoutcoder(filename)
        # class label for the whole image
        class_label = data_example['label']
        bbox = data_example['bbox']
        roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
        landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                    bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]

        dataset.append([image_data,class_label,roi,landmark])
        i = i+1
        if (i % 100 == 0):
            printStr = "\r the sample [{}] ".format(i)
            sys.stdout.write(printStr)
            sys.stdout.flush()

    return dataset


def gen_imdb(filename,net,type,shuffling):
    if os.path.exists(filename):
        os.remove(filename)
    # GET Dataset, and shuffling.
    dataset = __get_dataset(net=net, iterType=type)
    if shuffling:
        np.random.shuffle(dataset)
    
    fid = open(filename,'wb')
    pickle.dump(dataset, fid)
    fid.close()


def start(net,shuffling=False):

    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    #tfrecord name
    if net == 'pnet':
        imFileName = os.path.join(saveFolder, "pnet.imdb")
        gen_imdb(imFileName, net, 'all', shuffling)
    elif net in ['rnet', 'onet']:
        for n in ['pos', 'neg', 'part', 'landmark']:
            tfFileName = os.path.join(saveFolder, "%s.imdb"%(n))
            gen_imdb(tfFileName, net, n, shuffling)
    # Finally, write the labels file:
    print('\nFinished converting the MTCNN dataset!')






def parse_args():
    parser = argparse.ArgumentParser(description='Create imdb file...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args




if __name__ == "__main__":

    args = parse_args()
    stage = args.stage
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # set GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    start(stage, True)
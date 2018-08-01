#coding:utf-8
import sys
import numpy as np
import cv2
import os
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from util.common import IOU

def gen_hard_bbox_pnet(srcDataSet, srcAnnotations):
    srcDataSet = os.path.join(rootPath, srcDataSet)
    srcAnnotations = os.path.join(rootPath, srcAnnotations)
    saveFolder = os.path.join(rootPath, "tmp/data/pnet/")
    print(">>>>>> Gen hard samples for pnet...")
    typeName = ["pos", "neg", "part"]
    saveFiles = {}
    for tp in typeName:
        _saveFolder = os.path.join(saveFolder, tp)
        if not os.path.isdir(_saveFolder):
            os.makedirs(_saveFolder)
        saveFiles[tp] = open(os.path.join(saveFolder, "{}.txt".format(tp)), 'w')

    annotationsFile = open(srcAnnotations, "r")
    pIdx = 0 # positive
    nIdx = 0 # negative
    dIdx = 0 # dont care
    idx = 0
    for annotation in annotationsFile:
        annotation = annotation.strip().split(' ')
        # image path
        imPath = annotation[0]
        # boxed change to float type
        #box坐标转换为array数组  使用np处理数据
        bbox = list(map(float, annotation[1:]))
        # gt. each row mean bounding box
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        #load image
        file_abspath = os.path.join(srcDataSet, imPath +'.jpg')
        print(file_abspath)
        img = cv2.imread(file_abspath)

        idx += 1
        height, width, channel = img.shape  #注意宽高的位置

        # 1. NEG: random to crop negative sample image
        negNum = 0
        while negNum < 50:
            #负例图片大小
            neg_size = np.random.randint(12,min(width,height)/2)
            neg_topx = np.random.randint(0,width - neg_size) #保证剪切图片右边不超出原图边界
            neg_topy = np.random.randint(0,height - neg_size) #确保剪切图片上边界不超出原图上界

            crop_box = np.array([neg_topx,neg_topy,neg_topx + neg_size,neg_topy + neg_size])
            # random crop
            # cal iou and iou must below 0.3 for neg sample
            iou = IOU(crop_box, boxes)
            if np.max(iou) >= 0.3:
                continue
            # crop sample image
            crop_img = img[neg_topy:neg_topy+neg_size,neg_topx:neg_topx+neg_size,:]
            resize_img = cv2.resize(crop_img,(12,12))

            # now to save it
            save_file = os.path.join(saveFolder, "neg", "%s.jpg"%nIdx)
            saveFiles['neg'].write(save_file + ' 0\n')
            cv2.imwrite(save_file, resize_img)
            nIdx += 1
            negNum += 1

        #在有人脸区域生成部分负例　　正例　　部分人脸
        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            #bbox's width and height
            w, h = x2 - x1 + 1, y2 - y1 + 1
            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            # 2. NEG: random to crop sample image in bbox inside
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IOU(crop_box, boxes)
                if np.max(iou) >= 0.3:
                    continue
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(saveFolder, "neg", "%s.jpg"%nIdx)
                saveFiles['neg'].write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                nIdx += 1
            # 3. POS and PART
            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
                #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #计算的偏移量　　保证卷积的尺度不变性　不同尺度得到预测输出相同　　计算的是比例
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                #crop
                cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                #resize
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IOU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(saveFolder, "pos", "%s.jpg"%pIdx)
                    saveFiles['pos'].write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    pIdx += 1
                elif IOU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(saveFolder, "part", "%s.jpg"%dIdx)
                    saveFiles['part'].write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    dIdx += 1
        printStr = "\r[{}] pos: {}  neg: {}  part:{}".format(idx, pIdx, nIdx, dIdx)
        sys.stdout.write(printStr)
        sys.stdout.flush()
    for f in saveFiles.values():
        f.close()
    print('\n')


if __name__ == "__main__":
    gen_hard_bbox_pnet("dataset/WIDER_train/images/", "dataset/wider_face_train.txt")

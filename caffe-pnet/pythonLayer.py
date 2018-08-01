import sys
sys.path.append('/home/cmcc/caffe-master/python')
import cv2
import caffe
import numpy as np
import random
import cPickle as pickle
imdb_exit = True

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
		layer_params = yaml.load(self.param_str)
		self._batch_size = int(layer_params.get('batch_size', 256))
		self.net_size = layer_params.get('net_size', 12)
		self.source = layer_params.get('source', 'CSV')
		cls_list = ''
		cls_root = ''

        self.batch_loader = BatchLoader(cls_list,roi_list,pts_list,net_side,cls_root,roi_root,pts_root)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 1)
		top[2].reshape(self.batch_size, 4)
		top[3].reshape(self.batch_size, 10)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
		loss_task = random.randint(0,2)
        for itt in range(self.batch_size):
            im, label, roi, pts= self.batch_loader.load_next_image(loss_task)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
	    top[2].data[itt, ...] = roi
	    top[3].data[itt, ...] = pts
    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,cls_list,roi_list,pts_list,net_side,cls_root,roi_root,pts_root):
		self.mean = 128
        self.im_shape = net_side
        self.cls_root = cls_root
		self.roi_root = roi_root
		self.pts_root = pts_root
		self.roi_list = []
		self.cls_list = []
		self.pts_list = []
		print "Start Reading Classify Data into Memory..."
		
		fid = open('12/cls.imdb','r')
		self.cls_list = pickle.load(fid)
		fid.close()

		random.shuffle(self.cls_list)


    def load_next_image(self): 
		if self.cls_cur == len(self.cls_list):
			self.cls_cur = 0
			random.shuffle(self.cls_list)
		cur_data = self.cls_list[self.cls_cur]  # Get the image index
		im       = cur_data[0]
		label    = cur_data[1]
		roi      = [-1,-1,-1,-1]
		pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
		if random.choice([0,1])==1:
			im = cv2.flip(im,random.choice([-1,0,1]))
		self.cls_cur += 1
		return im, label, roi, pts


################################################################################
######################Regression Loss Layer By Python###########################
################################################################################
class regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	if bottom[0].count != bottom[1].count:
	    raise Exception("Input predict and groundTruth should have same dimension")
	roi = bottom[1].data
	self.valid_index = np.where(roi[:,0] != -1)[0]
	self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
	self.diff[...] = 0
	top[0].data[...] = 0
	if self.N != 0:
	    self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
            top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self,top,propagate_down,bottom):
	for i in range(2):
	    if not propagate_down[i] or self.N==0:
		continue
	    if i == 0:
		sign = 1
	    else:
		sign = -1
	    bottom[i].diff[...] = sign * self.diff / bottom[i].num
################################################################################
#############################Classify Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	top[0].reshape(len(bottom[1].data), 2,1,1)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]

class cls_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	top[0].reshape(len(bottom[1].data), 2)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]

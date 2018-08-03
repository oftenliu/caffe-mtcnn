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
		self.source = layer_params.get('source', 'tmp/data/pnet')


        self.batch_loader = BatchLoader(self.net_size,self.source)
        top[0].reshape(self.batch_size, 3, self.net_size, self.net_size)
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
    def __init__(net_size,source):
		self.mean = 128
        self.im_shape = net_size
		self.dataset = []

		print("Start Reading pnet Data into Memory...")
		
		fid = open(source,'r')
		self.dataset = pickle.load(fid)
		fid.close()

		random.shuffle(self.cls_list)


	def random_flip_images(image, label, landmark):
		# mirror
		if np.random.choice([0, 1]) > 0:
		# random flip
			if label == -2 or  label == 1:
				cv2.flip(image_batch[i], 1, image_batch[i])

				# pay attention: flip landmark
			if label == -2:
				landmark_ = landmark.reshape((-1, 2))
				landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
				landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
				landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
				landmark = landmark_.ravel()

		return image_batch, landmark_batch


    def load_next_image(self): 
		if self.cls_cur == len(self.dataset):
			self.cls_cur = 0
			random.shuffle(self.dataset)
		curdata = self.dataset[self.cls_cur]  # Get the image index
		im       = curdata[0]
		label    = curdata[1]
		roi      = [curdata[2],curdata[3],curdata[4],curdata[5]]
		landmark = [curdata[6],curdata[7],curdata[8],curdata[9],curdata[10],curdata[11],curdata[12],curdata[13],curdata[14],curdata[15]]
		random_flip_images(im,label,landmark)
		self.cls_cur += 1
		return im, label, landmark


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














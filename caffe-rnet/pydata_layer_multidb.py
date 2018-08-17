import multiprocessing
import cv2
import lmdb
import numpy as np
import sys
sys.path.append('/home/ulsee/often/caffe/python')
import yaml
import caffe
import atexit
from functools import reduce
from config import cfg

__authors__ = ['often(1992often@gmail.com)']

class MtcnnDataLayer(caffe.Layer):
    '''Custom Data Layer
    LayerOutput
      top[0]: image data
      top[1]: bbox target
      top[2]: landmark target
      top[3]: face data type / label, 0 for negatives, 1 for positives
                                      -1 for part faces, -2 for landmark faces

    Howto
      layer {
        name: "data"
        type: "Python"
        top: "data"
        top: "label"
        top: "bbox_target"
        top: "landmark_target"

        python_param {
          module: "layers.data_layer"
          layer: "FaceDataLayer"
        }
      }
    '''
    #self.data_size = [int(np.ceil(batch_size*ratio)) for ratio in data_ratio]
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.batch_size = int(layer_params.get('batch_size', 256))
        self.net_type = layer_params.get('net_type', 'pnet')
        self.net_size = cfg.NET_INPUT_SIZE[self.net_type]

    
        db_names_train = ['../tmp/data/%s/negdb'%self.net_type,
                          '../tmp/data/%s/posdb'%self.net_type,
                          '../tmp/data/%s/partdb'%self.net_type,
                          '../tmp/data/%s/landmarkdb'%self.net_type]

        self.queue_train = multiprocessing.Queue(32)
        batcher_train = MiniBatcher(db_names_train, self.batch_size, self.net_type)

        batcher_train.set_queue(self.queue_train)
        batcher_train.start()

        def cleanup():
            batcher_train.terminate()
            batcher_train.join()
        
        atexit.register(cleanup)

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, 3, self.net_size, self.net_size) #image
        top[1].reshape(self.batch_size) #label
        top[2].reshape(self.batch_size, 4)  #bbox
        top[3].reshape(self.batch_size, 10) #landmark


    def forward(self, bottom, top):
        minibatch = self.get_minibacth()
        top[0].data[...] = minibatch['data']
        top[1].data[...] = minibatch['label']
        # face data
        top[2].data[...] = minibatch['bbox_target']
        top[3].data[...] = minibatch['landmark_target']


    def backward(self, bottom, top):
        pass

    def get_minibacth(self):
        minibatch = self.queue_train.get()
        return minibatch


class MiniBatcher(multiprocessing.Process):
    '''generate minibatch
    given a queue, put (negatives, positives, part faces, landmark faces) = (n1, n2, n3, n4)
    '''

    def __init__(self, db_names, batch_size, net_type):
        '''order: negatives, positives, part faces, landmark faces
          net_type: pnet rnet onet
        '''
        super(MiniBatcher, self).__init__()
        self.batch_size = batch_size
        self.start_pos = [0 for _ in range(4)]
        self.net_type = net_type
        self.db_names = db_names
        self.db = [lmdb.open(db_name) for db_name in db_names]
        self.tnx = [db.begin() for db in self.db]
        self.db_size = [int(tnx.get('size'.encode())) for tnx in self.tnx]
        print("db_size[0]=%d db_size[1]=%d db_size[2]=%d db_size[3]=%d"%(self.db_size[0],self.db_size[1],self.db_size[2],self.db_size[3]))
        data_ratio = reduce(lambda x,y:x+y,cfg.DATA_RATIO[self.net_type])
        self.data_size = [int(np.ceil(batch_size*ratio/data_ratio)) for ratio in cfg.DATA_RATIO[self.net_type]]  #向上取整
        self.data_size[0] = self.batch_size - self.data_size[1] - self.data_size[2] - self.data_size[3]
        print("neg=%d pos=%d part=%d landmark=%d"%(self.db_size[0],self.db_size[1],self.db_size[2],self.db_size[3]))
        print("neg_data_size[0]=%d pos_data_size[1]=%d part_data_size[2]=%d landmark_data_size[3]=%d"%(self.data_size[0],self.data_size[1],self.data_size[2],self.data_size[3]))
        self.net_size = cfg.NET_INPUT_SIZE[self.net_type]

    def __del__(self):
        for tnx in self.tnx:
            tnx.abort()
        for db in self.db:
            db.close()

    def set_queue(self, queue):
        self.queue = queue

    def get_size(self):
        return self.db_size


    def random_flip_image(self,image,bbox, landmark):
      # mirror
      if np.random.choice([0, 1]) > 2:
          # random flip

          cv2.flip(image, 1, image)

          bbox[0], bbox[2] = -bbox[2], -bbox[0]
          # pay attention: flip landmark

          landmark_ = landmark.reshape((-1, 2))
          landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
          landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
          landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
          landmark = landmark_.ravel()

      return image,bbox, landmark


    def run(self):
        intpu_size = self.net_size
        data_shape = (3,intpu_size, intpu_size)
        bbox_shape = (4,)
        landmark_shape = (10,)
        batch_size = self.batch_size
        while True:
            data = np.zeros((batch_size, 3, intpu_size, intpu_size), dtype=np.float32)
            bbox_target = np.zeros((batch_size, 4), dtype=np.float32)
            landmark_target = np.zeros((batch_size, 10), dtype=np.float32)
            label = np.zeros(batch_size, dtype=np.int32)

            start = self.start_pos
            end = [start[i] + self.data_size[i] for i in range(4)]
            for i in range(4):
                if end[i] > self.db_size[i]:
                  end[i] -= self.db_size[i]
                  start[i] = end[i]
                  end[i] = start[i] + self.data_size[i]

            
            idx = 0
            #print("\n\n\n==============================\n\n\n============================\n\n\n\n")
            #print("here start[0] = [%d], end[0] = [%d]"%(start[0],end[0]))
            #print("\n\n\n==============================\n\n\n============================\n\n\n\n")
            # negatives

            for i in range(start[0], end[0]):
                data_key = '%08d_data'%i
                label_key = '%08d_label'%i
                bbox_key = '%08d_bbox'%i
                landmark_key = '%08d_landmark'%i

                data[idx] = np.fromstring(self.tnx[0].get(data_key.encode()), dtype=np.float32).reshape(data_shape)
                label[idx] = np.fromstring(self.tnx[0].get(label_key.encode()), dtype=np.int32)
                #print(label[idx])
                bbox_target[idx] = np.fromstring(self.tnx[0].get(bbox_key.encode()), dtype=np.float32).reshape(bbox_shape)
                landmark_target[idx] = np.fromstring(self.tnx[0].get(landmark_key.encode()), dtype=np.float32).reshape(landmark_shape)
                
                idx += 1
            # positives
            for i in range(start[1], end[1]):
                data_key = '%08d_data'%i
                label_key = '%08d_label'%i
                bbox_key = '%08d_bbox'%i
                landmark_key = '%08d_landmark'%i

                data[idx] = np.fromstring(self.tnx[1].get(data_key.encode()), dtype=np.float32).reshape(data_shape)
                
                label[idx] = np.fromstring(self.tnx[1].get(label_key.encode()), dtype=np.int32)
                #print(label[idx])
                bbox_target[idx] = np.fromstring(self.tnx[1].get(bbox_key.encode()), dtype=np.float32).reshape(bbox_shape)
                landmark_target[idx] = np.fromstring(self.tnx[1].get(landmark_key.encode()), dtype=np.float32).reshape(landmark_shape)
                data[idx],bbox_target[idx],landmark_target[idx] = self.random_flip_image(data[idx],bbox_target[idx],landmark_target[idx])
                idx += 1
            # part faces
            #print(data[0])
            for i in range(start[2], end[2]):
                data_key = '%08d_data'%i
                label_key = '%08d_label'%i
                bbox_key = '%08d_bbox'%i
                landmark_key = '%08d_landmark'%i

                data[idx] = np.fromstring(self.tnx[2].get(data_key.encode()), dtype=np.float32).reshape(data_shape)
                label[idx] = np.fromstring(self.tnx[2].get(label_key.encode()), dtype=np.int32)
                #print(label[idx])
                bbox_target[idx] = np.fromstring(self.tnx[2].get(bbox_key.encode()), dtype=np.float32).reshape(bbox_shape)
                landmark_target[idx] = np.fromstring(self.tnx[2].get(landmark_key.encode()), dtype=np.float32).reshape(landmark_shape)
                idx += 1
            # landmark faces
            for i in range(start[3], end[3]):
                data_key = '%08d_data'%i
                label_key = '%08d_label'%i
                bbox_key = '%08d_bbox'%i
                landmark_key = '%08d_landmark'%i

                data[idx] = np.fromstring(self.tnx[3].get(data_key.encode()), dtype=np.float32).reshape(data_shape)
                label[idx] = np.fromstring(self.tnx[3].get(label_key.encode()), dtype=np.int32)
                #print(label[idx])
                bbox_target[idx] = np.fromstring(self.tnx[3].get(bbox_key.encode()), dtype=np.float32).reshape(bbox_shape)
                landmark_target[idx] = np.fromstring(self.tnx[3].get(landmark_key.encode()), dtype=np.float32).reshape(landmark_shape)
                data[idx],bbox_target[idx],landmark_target[idx] = self.random_flip_image(data[idx],bbox_target[idx],landmark_target[idx])
                idx += 1

            self.start_pos = end

            minibatch = {'data': data,
                        'bbox_target': bbox_target,
                        'landmark_target': landmark_target,
                        'label': label}
            self.queue.put(minibatch)


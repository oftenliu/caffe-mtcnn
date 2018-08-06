import multiprocessing
import cv2
import lmdb
import numpy as np
import sys
sys.path.append('/home/ulsee/often/caffe/python')
import json
import caffe
import atexit
from config import cfg

class FaceDataLayer(caffe.Layer):
    '''Custom Data Layer
    LayerOutput
      top[0]: image data
      top[1]: bbox target
      top[2]: landmark target
      top[3]: face data type / label, 0 for negatives, 1 for positives
                                      2 for part faces, 3 for landmark faces

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

    '''
        net_type = args.net
        self.net_type = net_type
        input_size = cfg.NET_INPUT_SIZE[net_type]
        db_names_train = ['data/%snet_negative_train'%net_type,
                          'data/%snet_positive_train'%net_type,
                          'data/%snet_part_train'%net_type,
                          'data/%snet_landmark_train'%net_type]
        db_names_test = ['data/%snet_negative_val'%net_type,
                        'data/%snet_positive_val'%net_type,
                        'data/%snet_part_val'%net_type,
                        'data/%snet_landmark_val'%net_type]
        base_size = args.size
        ns = [r*base_size for r in cfg.DATA_RATIO[net_type]]
        # batcher setup
        batcher_train = MiniBatcher(db_names_train, ns, net_type)
        batcher_test = MiniBatcher(db_names_test, ns, net_type)
        # data queue setup
        queue_train = multiprocessing.Queue(32)
        queue_test = multiprocessing.Queue(32)
        batcher_train.set_queue(queue_train)
        batcher_test.set_queue(queue_test)
        pos_ratio, part_ratio, landmark_ratio, neg_ratio = 1.0 / 6, 1.0 / 6, 1.0 / 6, 3.0 / 6
    '''
    #self.data_size = [int(np.ceil(batch_size*ratio)) for ratio in data_ratio]
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.batch_size = int(layer_params.get('batch_size', 256))
        self.net_type = layer_params.get('net_type', 'pnet')
        self.net_size = cfg.NET_INPUT_SIZE[self.net_type]
        self.source = layer_params.get('source', 'tmp/data/pnet')


    
        db_names_train = ['tmp/data/%s/posdb'%self.net_type,
                          'tmp/data/%s/negdb'%self.net_type,
                          'tmp/data/%s/partdb'%self.net_type,
                          'tmp/data/%s/landmarkdb'%self.net_type]

        self.queue_train = multiprocessing.Queue(32)
        batcher_train = MiniBatcher(db_names_train, self.batch_size, self.net_type)

        batcher_train.set_queue(self.queue_train)
        batcher_train.start()

        def cleanup():
            batcher_train.terminate()
            batcher_train.join()
        
        atexit.register(cleanup)

    def reshape(self, bottom, top):
        top[0].reshape(self.n, 3, self.net_input_size, self.net_input_size)
        top[1].reshape(self.n, 4)
        top[2].reshape(self.n, 10)
        top[3].reshape(self.n)

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
        self.start = [0 for _ in range(4)]
        self.net_type = net_type
        self.db_names = db_names
        self.db = [lmdb.open(db_name) for db_name in db_names]
        self.tnx = [db.begin() for db in self.db]
        self.db_size = [int(tnx.get('size')) for tnx in self.tnx]
        
        self.data_size = [int(np.ceil(batch_size*ratio)) for ratio in cfg.DATA_RATIO[self.net_type]]

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
      if np.random.choice([0, 1]) > 0:
          # random flip

          cv2.flip(image, 1, image)

          bbox[0], bbox[2] = 1-bbox[2], 1-bbox[0]
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

          start = self.start
          end = [start[i] + self.data_size[i] for i in range(4)]
          for i in range(4):
            if end[i] > self.db_size[i]:
              end[i] -= self.db_size[i]
              start[i] = end[i]
              end[i] = start[i] + self.data_size[i]

          idx = 0
          # negatives
          for i in range(start[0], end[0]):
            data_key = '%08d_data'%i
            label_key = '%08d_label'%i
            bbox_key = '%08d_bbox'%i
            landmark_key = '%08d_landmark'%i

            data[idx] = np.fromstring(self.tnx[0].get(data_key), dtype=np.uint8).reshape(data_shape)
            label[idx] = np.fromstring(self.tnx[0].get(label_key), dtype=np.int32)
            bbox_target[idx] = np.fromstring(self.tnx[0].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
            landmark_target[idx] = np.fromstring(self.tnx[0].get(landmark_key), dtype=np.float32).reshape(landmark_shape)
            
            idx += 1
          # positives
          for i in range(start[1], end[1]):
            data_key = '%08d_data'%i
            label_key = '%08d_label'%i
            bbox_key = '%08d_bbox'%i
            landmark_key = '%08d_landmark'%i

            data[idx] = np.fromstring(self.tnx[0].get(data_key), dtype=np.uint8).reshape(data_shape)
            label[idx] = np.fromstring(self.tnx[0].get(label_key), dtype=np.int32)
            bbox_target[idx] = np.fromstring(self.tnx[0].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
            landmark_target[idx] = np.fromstring(self.tnx[0].get(landmark_key), dtype=np.float32).reshape(landmark_shape)
            data[idx],bbox_target[idx],landmark_target[idx] = random_flip_image(data[idx],bbox_target[idx],landmark_target[idx])
            idx += 1
          # part faces
          for i in range(start[2], end[2]):
            data_key = '%08d_data'%i
            label_key = '%08d_label'%i
            bbox_key = '%08d_bbox'%i
            landmark_key = '%08d_landmark'%i

            data[idx] = np.fromstring(self.tnx[0].get(data_key), dtype=np.uint8).reshape(data_shape)
            label[idx] = np.fromstring(self.tnx[0].get(label_key), dtype=np.int32)
            bbox_target[idx] = np.fromstring(self.tnx[0].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
            landmark_target[idx] = np.fromstring(self.tnx[0].get(landmark_key), dtype=np.float32).reshape(landmark_shape)
            idx += 1
          # landmark faces
          for i in range(start[3], end[3]):
            data_key = '%08d_data'%i
            label_key = '%08d_label'%i
            bbox_key = '%08d_bbox'%i
            landmark_key = '%08d_landmark'%i

            data[idx] = np.fromstring(self.tnx[0].get(data_key), dtype=np.uint8).reshape(data_shape)
            label[idx] = np.fromstring(self.tnx[0].get(label_key), dtype=np.int32)
            bbox_target[idx] = np.fromstring(self.tnx[0].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
            landmark_target[idx] = np.fromstring(self.tnx[0].get(landmark_key), dtype=np.float32).reshape(landmark_shape)
            data[idx],bbox_target[idx],landmark_target[idx] = random_flip_image(data[idx],bbox_target[idx],landmark_target[idx])
            idx += 1

          self.start = end

          minibatch = {'data': data,
                      'bbox_target': bbox_target,
                      'landmark_target': landmark_target,
                      'label': label}
          self.queue.put(minibatch)
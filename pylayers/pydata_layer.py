import multiprocessing
import cv2
import lmdb
import numpy as np
sys.path.append('/home/ulsee/often/caffe/python')
import json
import caffe



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


  def set_batch_num(self, n1, n2, n3, n4):
    '''set data type number
    n1 for negatives, n2 for positives, n3 for part faces, n4 for landmark faces
    net_input_size for network input size (width, height)
    '''
    self.n1 = n1
    self.n2 = n2
    self.n3 = n3
    self.n4 = n4
    self.n = n1 + n2 + n3 + n4
    self.net_input_size = cfg.NET_INPUT_SIZE[cfg.NET_TYPE]

  def set_data_queue(self, queue):
    '''the queue should put a minibatch with size of (negatives, positives, part faces, landmark faces) =
    (n1, n2, n3, n4) in a dict
    '''
    self.data_queue = queue

  def setup(self, bottom, top):
    layer_params = yaml.load(self.param_str)
		self._batch_size = int(layer_params.get('batch_size', 256))
		self.net_size = layer_params.get('net_size', 12)
		self.source = layer_params.get('source', 'tmp/data/pnet')

    self.n1 = 1
    self.n2 = 1
    self.n3 = 1
    self.n4 = 1
    self.n = 4
    self.net_input_size = cfg.NET_INPUT_SIZE[cfg.NET_TYPE]
    self.reshape(bottom, top)

  def reshape(self, bottom, top):
    top[0].reshape(self.n, 3, self.net_input_size, self.net_input_size)
    top[1].reshape(self.n, 4)
    top[2].reshape(self.n, 10)
    top[3].reshape(self.n)

  def forward(self, bottom, top):
    minibatch = self._get_minibacth()
    # face data
    top[0].data[...] = minibatch['data']
    top[1].data[...] = minibatch['bbox_target']
    top[2].data[...] = minibatch['landmark_target']
    top[3].data[...] = minibatch['label']

  def backward(self, bottom, top):
    pass

  def _get_minibacth(self):
    minibatch = self.data_queue.get()
    return minibatch


class MiniBatcher(multiprocessing.Process):
  '''generate minibatch
  given a queue, put (negatives, positives, part faces, landmark faces) = (n1, n2, n3, n4)
  '''

  def __init__(self, db_names, batch_size, data_ratio, net_type):
    '''order: negatives, positives, part faces, landmark faces
       net_type: pnet rnet onet
    '''
    super(MiniBatcher, self).__init__()
    self.data_size = [int(np.ceil(batch_size*ratio)) for ratio in data_ratio]
    self.batch_size = batch_size
    self._start = [0 for _ in range(4)]
    self.net_type = net_type
    self.db_names = db_names
    self.db = [lmdb.open(db_name) for db_name in db_names]
    self.tnx = [db.begin() for db in self.db]
    self.db_size = [int(tnx.get('size')) for tnx in self.tnx]
    if net_type == 'pnet':
        self.intpu_size = 12

  def __del__(self):
    for tnx in self.tnx:
      tnx.abort()
    for db in self.db:
      db.close()

  def set_queue(self, queue):
    self.queue = queue

  def get_size(self):
    return self.db_size

  def _make_transform(self, data, bbox=None, landmark=None):
    # gray scale
    if np.random.rand() < cfg.GRAY_PROB:
      gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
      data[:, :, 0] = gray
      data[:, :, 1] = gray
      data[:, :, 2] = gray
    # flip
    if np.random.rand() < cfg.FLIP_PROB:
      data = data[:, ::-1, :]
      if bbox is not None:
        # [dx1 dy1 dx2 dy2] --> [-dx2 dy1 -dx1 dy2]
        bbox[0], bbox[2] = -bbox[2], -bbox[0]
      if landmark is not None:
        landmark1 = landmark.reshape((-1, 2))
        # x --> 1 - x
        landmark1[:, 0] = 1 - landmark1[:, 0]
        landmark1[0], landmark1[1] = landmark1[1].copy(), landmark1[0].copy()
        landmark1[3], landmark1[4] = landmark1[4].copy(), landmark1[3].copy()
        landmark = landmark1.reshape(-1)
    data = data.transpose((2, 0, 1))
    return data, bbox, landmark

  def run(self):
    intpu_size = self.intpu_size
    data_shape = (intpu_size, intpu_size, 3)
    bbox_shape = (4,)
    landmark_shape = (10,)
    batch_size = self.batch_size
    while True:
      data = np.zeros((batch_size, 3, intpu_size, intpu_size), dtype=np.float32)
      bbox_target = np.zeros((batch_size, 4), dtype=np.float32)
      landmark_target = np.zeros((batch_size, 10), dtype=np.float32)
      label = np.zeros(batch_size, dtype=np.float32)

      start = self._start
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
        _data = np.fromstring(self.tnx[0].get(data_key), dtype=np.uint8).reshape(data_shape)
        data[idx], _1, _2 = self._make_transform(_data)
        idx += 1
      # positives
      for i in range(start[1], end[1]):
        data_key = '%08d_data'%i
        bbox_key = '%08d_bbox'%i
        _data = np.fromstring(self.tnx[1].get(data_key), dtype=np.uint8).reshape(data_shape)
        _bbox_target = np.fromstring(self.tnx[1].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
        data[idx], bbox_target[idx], _ = self._make_transform(_data, _bbox_target)
        idx += 1
      # part faces
      for i in range(start[2], end[2]):
        data_key = '%08d_data'%i
        bbox_key = '%08d_bbox'%i
        _data = np.fromstring(self.tnx[2].get(data_key), dtype=np.uint8).reshape(data_shape)
        _bbox_target = np.fromstring(self.tnx[2].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
        data[idx], bbox_target[idx], _ = self._make_transform(_data, _bbox_target)
        idx += 1
      # landmark faces
      for i in range(start[3], end[3]):
        data_key = '%08d_data'%i
        landmark_key = '%08d_landmark'%i
        _data = np.fromstring(self.tnx[3].get(data_key), dtype=np.uint8).reshape(data_shape)
        _landmark_target = np.fromstring(self.tnx[3].get(landmark_key), dtype=np.float32).reshape(landmark_shape)
        data[idx], _, landmark_target[idx] = self._make_transform(_data, None, _landmark_target)
        idx += 1
      # label
      label[:self.ns[0]] = 0
      label[self.ns[0]: self.ns[0]+self.ns[1]] = 1
      label[self.ns[0]+self.ns[1]: self.ns[0]+self.ns[1]+self.ns[2]] = 2
      label[self.ns[0]+self.ns[1]+self.ns[2]:] = 3

      self._start = end
      data = (data - 128) / 128 # simple normalization
      minibatch = {'data': data,
                   'bbox_target': bbox_target,
                   'landmark_target': landmark_target,
                   'label': label}
      self.queue.put(minibatch)
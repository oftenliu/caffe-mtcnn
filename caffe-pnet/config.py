import os
from easydict import EasyDict


cfg = EasyDict()

# data directories
cfg.Data_DIR = 'data/'
cfg.CelebA_DIR = 'data/CelebA/'
cfg.WIDER_DIR = 'data/WIDER/'
cfg.FDDB_DIR = 'data/fddb/'

cfg.NET_INPUT_SIZE = {'pnet': 12, 'rnet': 24, 'onet': 48}




# training data ratio in a minibatch, [negative, positive, part, landmark]
cfg.DATA_RATIO = {
  'pnet': [3, 1, 1, 2],
  'rnet': [3, 1, 1, 2],
  'onet': [3, 2, 1, 2],
}


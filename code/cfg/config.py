#import os
import os.path as osp
#import numpy as np

from easydict import EasyDict as edict

__C = edict()


cfg = __C

# Maximum margin to pick up samples around objects
__C.D = 40


# strategy3 margin
__C.NEG3_MARGIN = 15


# margin size for candidate pixels
__C.D_MARGIN = 1

# Ratio factor for determine the distance among pixels
__C.RATIO_FACTOR = 6.


# INPUT DataSet Attribute
__C.TXT_PATH = '/home/yuanjial/NeuralNetwork/FilePreprocess/coco/train.txt'
__C.BASE_DIR = '/home/yuanjial/DataSet/COCO/coco2014_train'

__C.IMG_DIR = 'Images'
__C.IMG_EXT = '.jpg'

__C.INSTANCEANN_DIR = 'InstanceAnn'
__C.GT_EXT = '.png'

# directory to store converted tiff files
__C.OUT_PATH = 'converted'


# Number of positive sampels
__C.N_POS = 4

# Number of negative samples
__C.N_NEG = 20

# Number of pairs
__C.N_PAIRS = 6

# Energy scale in distrance transform
__C.ENERGY_SCALE = 4

# Training phase configures
__C.TRAIN = edict()


# Test Configures
__C.TEST = edict()

# Project Root
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))


# GPU ID
__C.GPU_ID = 0

# Small Number
__C.EPS = 1e-14







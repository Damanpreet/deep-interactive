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

# Training image extension
__C.IMG_EXT = '.jpg'

# Testing image extension
__C.GT_EXT = '.png'

# Data Root
__C.DATA_DIR = '/home/yuanjial/DataSet/PASCAL_aug'

# Trianing image data directory name
__C.IMG_DIR = osp.join(__C.DATA_DIR, 'JPEGImages')

# Benchmark Directory
__C.BENCHMARK_DIR = 'PASCAL'


# Testing image data directory name
__C.GT_DIR = osp.join(__C.DATA_DIR, 'SegmentationObjectFilledDenseCRF')

# Number of positive sampels
__C.N_POS = 5

# Number of negative samples
__C.N_NEG = 20

# Number of pairs
__C.N_PAIRS = 10

# Energy scale in distrance transform
__C.ENERGY_SCALE = 4

# Training phase configures
__C.TRAIN = edict()


# Test Configures
__C.TEST = edict()

# Project Root
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))


# Models Root
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models'))


# Converted images directory
__C.NEW_DIR = osp.abspath(osp.join(__C.DATA_DIR, 'converted_images'))


# GPU ID
__C.GPU_ID = 0

# Small Number
__C.EPS = 1e-14







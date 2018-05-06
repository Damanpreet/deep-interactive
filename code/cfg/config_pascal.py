#import os
import os.path as osp
from easydict import EasyDict as edict

__C = edict()


cfg = __C

# Maximum margin to pick up samples around objects
__C.D = 80


# strategy3 margin
__C.NEG3_MARGIN = 80


# margin size for candidate pixels
__C.D_MARGIN = 1

# Ratio factor for determine the distance among pixels
__C.RATIO_FACTOR = 6.


# INPUT DataSet Attribute
# __C.TXT_PATH = '/home/yuanjial/NeuralNetwork/FilePreprocess/coco/train.txt'
# __C.BASE_DIR = '/home/yuanjial/DataSet/COCO/coco2014_train'

__C.TXT_PATH = '/home/yuanjial/NeuralNetwork/FilePreprocess/pascal/val.txt'
__C.BASE_DIR = '/media/zhouzh/LargeDisk/yjl_dataset/PASCAL_aug'

__C.IMG_DIR = 'JPEGImages'
__C.IMG_EXT = '.jpg'

__C.INSTANCEANN_DIR = 'SegmentationObjectFilledDenseCRF'
__C.GT_EXT = '.png'

# directory to store converted tiff files
__C.OUT_PATH = 'PASCAL/converted'
__C.OUT_FNAME = 'val_converted_pascal.txt'

# Number of positive sampels
__C.N_POS = 2

# Number of negative samples
__C.N_NEG = 3

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







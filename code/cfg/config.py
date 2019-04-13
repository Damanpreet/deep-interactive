#import os
import os.path as osp
from easydict import EasyDict as edict

__C = edict()


cfg = __C

# Maximum margin to pick up samples around objects
__C.D = 20


# strategy3 margin
__C.NEG3_MARGIN = 10


# margin size for candidate pixels
__C.D_MARGIN = 10

# Ratio factor for determine the distance among pixels
__C.RATIO_FACTOR = 2.


# INPUT DataSet Attribute
__C.DATASET  = 'train'
__C.TXT_PATH = './dataset/pascal/'+__C.DATASET +'.txt'
__C.BASE_DIR = './dataset/pascal/PASCAL/'

__C.IMG_DIR = osp.join(__C.BASE_DIR, 'sourceData/JPEGImages')
__C.IMG_EXT = '.jpg'

#__C.INSTANCEANN_DIR = osp.join(__C.BASE_DIR, 'sourceData/Merge_VOC_SBD/inst/')
__C.INSTANCEANN_DIR = osp.join(__C.BASE_DIR, 'sourceData/CRF-Refine-Ann/SegmentationObjectFilledDenseCRF/')
__C.GT_EXT = '.png'

# directory to store converted tiff files
__C.OUT_PATH = osp.join(__C.BASE_DIR, 'sample_PN_converted')
__C.OUT_TXT_PATH = osp.join(__C.OUT_PATH, __C.DATASET+'_converted_pascal.txt')

# Number of positive sampels
__C.N_POS = 5

# Number of negative samples
__C.N_NEG = 30

# Number of pairs
__C.N_PAIRS = 3

# Energy scale in distrance transform
__C.ENERGY_SCALE = 6

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







import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# image path
__C.BASE_PATH = './dataset/pascal/PASCAL/'

__C.RGB_PATH  = osp.join(__C.BASE_PATH, 'sourceData/JPEGImages/')
__C.RGB_EXT   = '.jpg'

__C.PNSAMPLE_PATH  = osp.join(__C.BASE_PATH, 'sample_PN_converted/data/')
__C.PNSAMPLE_EXT   = '_pnsamples.tif'

__C.GT_PATH = osp.join(__C.BASE_PATH, 'sample_PN_converted/labels/')
__C.GT_EXT  = '_label.tif'


__C.OUT_PATH = osp.join('./output_pos_on_edge/')

# about dataset
__C.VAL_LIST         = './dataset/pascal/PASCAL/sample_PN_converted/val_converted_pascal.txt'
__C.TRAIN_LIST       = './dataset/pascal/PASCAL/sample_PN_converted/train_converted_pascal.txt'


# about process.
__C.ONLY_POS         = False
__C.INPUT_SIZE       = '321,321'
__C.num_classes      = 1


#
__C.IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434, 131.0, 131.0), dtype=np.float32)





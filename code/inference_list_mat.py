"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
# from datetime import datetime
import os
# import sys
# import time
import pdb
import cv2
import scipy.io
from matplotlib import image , cm
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel #, decode_labels, prepare_label
import denseCRF_infer3 as dcrf3
import instance_falseColor as instanceRGB

ONLY_POS = False
DATA_DIRECTORY = '/home/yuanjial/DataSet/PASCAL_aug/'
DATA_LIST_NAME = './dataImg/val_smpl.txt'
SAVE_DIR = './output/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434, 156.042324, 156.523433), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--img_path", type=str, default=DATA_DIRECTORY,
                        help="Path to the RGB image file.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def read_image_list():
    '''
     output: rgbList: list of path to rgbimages
             pnList: list of path to sample pos/neg map packages.
             labelList: list of path to labels
    '''
    f = open(DATA_LIST_NAME, 'r')
    imgList, pnList, labelList = [], [], []
    for line in f:
        tmp = line.strip("\n").split('')
        rgbImg, pnImg, labelImg = tmp[0], tmp[1], tmp[-1]

        imgList.append(DATA_DIRECTORY + rgbImg)
        pnList.append(DATA_DIRECTORY + pnImg)
        labelList.append(DATA_DIRECTORY + labelImg)

    return imgList, pnList, labelList


def loadImages(rgbPath, pnPath, labelPath=''):
    ''' load and pack data as a batch
     Args:
       rgbPath:  path to the rgb image file.
       pnPath:   path to the positive/negtive energy map package.
       labelPath:path to the Ground Truth label file.

     Returns:
        batch of input data for neural network,has shape(batch, ht, wd, 5)%[rgbImg, pos, neg].
        batch of label data
    '''
    img = cv2.imread(rgbPath)
    pnData = scipy.io.loadmat(pnPath)['objsInfo']

    classIds, images = [], []
    for ele in pnData:
        objData = ele[0][0][0]
        posMap = objData[0][..., np.newaxis]
        negMap = objData[1][..., np.newaxis]

        packData = np.concatenate((img, posMap, negMap), axis=2)
        packData = packData - IMG_MEAN
        images.append(packData)
        classIds.append(objData[2][0,0])

    return np.array(images), classIds, len(pnData)

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # Prepare image.
    imgList, smplList = read_image_list()
    # Create network.
    with tf.device('/gpu:1'):
        input_imgs = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_img')
        net = DeepLabResNetModel({'data': input_imgs}, is_training=False)

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output = net.layers['fc1_voc12']
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(input_imgs)[1:3,])
        # pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print('before run inference run.\n')

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for rgbPath, smplPath in zip(imgList, smplList):
        #rgbPath = '/home/yuanjial/DataSet/PASCAL_aug/JPEGImages/2007_001526.jpg'
        #smplPath = '/home/yuanjial/DataSet/PASCAL_aug/pos_neg_Map/2007_001526_pnMap.mat'
        save_fname = rgbPath.split('/')[-1]
        save_fname = save_fname.split('.')[0]

        # Perform inference.
        #rgbPath = '/home/yuanjial/DataSet/PASCAL_aug/JPEGImages/2007_001377.jpg'
        #smplPath = '/home/yuanjial/DataSet/PASCAL_aug/pos_neg_Map/2007_001377_pnMap.mat'
        inData, classList,  dataNum = loadImages(rgbPath, smplPath)
        if(inData.shape[-1] != 5):
            continue;

        preds = sess.run(raw_output_up, feed_dict={input_imgs:inData})

        for k in range(preds.shape[0]):
            img  = preds[k].squeeze()

            # msk = decode_labels(preds)
            # im = Image.fromarray(msk[0])
            # cv2.imwrite(args.save_dir+'mask.png', preds)
            image.imsave(args.save_dir+ save_fname+'_'+str(k)+'.png', img, cmap = cm.gray, vmin=0, vmax=255 )
            image.imsave(args.save_dir+ save_fname+'_color_'+str(k)+'.png', img) #, cmap = cm.gray, vmin=0, vmax=255 )

        fineInstance, coarseInstance = dcrf3.denseCRF_refine(save_fname, preds.shape[0])
        fineRGB = instanceRGB.enc_falseColorInstanceImage(fineInstance, classList)
        coarseRGB = instanceRGB.enc_falseColorInstanceImage(coarseInstance, classList)
        #pdb.set_trace()
        scipy.io.savemat(args.save_dir+save_fname+'.mat', {'label': fineInstance, 'classList':classList})
        image.imsave(args.save_dir + save_fname+ '_fine.png', fineRGB)
        image.imsave(args.save_dir + save_fname+ '_coarse.png', coarseRGB)
        print('The output file {} has been saved to {}'.format(save_fname, args.save_dir))


if __name__ == '__main__':
    main()

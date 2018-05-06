"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
import os
import time
# import pdb
import cv2
from tifffile import imread as tiff_imread

from matplotlib import image # , cm
import tensorflow as tf
import numpy as np
from deeplab_resnet import DeepLabResNetModel #, decode_labels, prepare_label

ONLY_POS = False
DATA_DIRECTORY = ''
DATA_LIST_PATH  = '/media/zhouzh/LargeDisk/yjl_dataset/CVPPP/CVPPP/converted/val_converted_cvppp.txt'
SAVE_DIR = './output_cvppp/'
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
    f = open(DATA_LIST_PATH, 'r')
    imgList, pnList, labelList = [], [], []
    for line in f:
        rgbImg, pnImg, labelImg = line.strip("\n").split(' ')

        imgList.append(DATA_DIRECTORY + rgbImg)
        pnList.append(DATA_DIRECTORY + pnImg)
        labelList.append(DATA_DIRECTORY + labelImg)

    return imgList, pnList, labelList



def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # Prepare image.
    imgList, pnList, labelList = read_image_list()
    ch = 4 if ONLY_POS else 5

    # Create network.
    with tf.device('/cpu:0'):
        input_imgs = tf.placeholder(tf.float32, shape=[None, None,None, ch], name='input_img')
        net = DeepLabResNetModel({'data': input_imgs}, is_training=False)

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output     = net.layers['fc1_voc12']
        raw_output_up  = tf.image.resize_bilinear(raw_output, tf.shape(input_imgs)[1:3,])
        preds          = tf.greater_equal(tf.nn.softmax(raw_output_up), 0.5)
        preds          = tf.cast(preds, tf.uint8)
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

    import pdb
    pdb.set_trace()
    for rgbPath, pnPath, labelPath in zip(imgList, pnList, labelList):
        save_fname = pnPath.split('/')[-1]
        save_fname = save_fname.split('.')[0]

        # Perform inference.
        rgbimg = cv2.imread(rgbPath)
        pnmaps = tiff_imread(pnPath)
        pnmaps = pnmaps[0, ...] if ONLY_POS else np.transpose(pnmaps, [1,2,0])
        pnmaps = pnmaps[..., np.newaxis] if ONLY_POS else pnmaps

        start_time = time.time();
        inData = np.concatenate((rgbimg, pnmaps), axis=2)
        inData = np.float32(inData)-IMG_MEAN[0:4] if ONLY_POS else np.float32(inData)-IMG_MEAN
        if(inData.shape[-1] != 5):
            continue;

        #rOutput, preds = sess.run([raw_output_up, preds], feed_dict={input_imgs:inData[np.newaxis, ...]})
        rOutput = sess.run(raw_output_up, feed_dict={input_imgs:inData[np.newaxis, ...]})

        #rOutput = 1/(1+np.exp(-rOutput))*255
        #img  = np.concatenate((rOutput[0], pnmaps), axis=2)
        img = rOutput.squeeze()

        # image.imsave(args.save_dir+ save_fname +'.png', preds.squeeze()*200, cmap = cm.gray, vmin=0, vmax=255 )
        image.imsave(args.save_dir+ save_fname +'_color.png', img) #, cmap = cm.gray, vmin=0, vmax=255 )
        duration = time.time()-start_time

        print('The output file {} has been saved to {}. -- time: {:.3f} sec/({:d}, {:d})'.format(save_fname, args.save_dir, duration, img.shape[0], img.shape[1]))


if __name__ == '__main__':
    main()

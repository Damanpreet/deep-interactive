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
from matplotlib import image #, cm
from tifffile import imread as tiff_imread
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel #, decode_labels, prepare_label

SAVE_DIR = './output_test/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434, 156.042324, 156.523433), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("pnmap_path", type=str,
                        help="Path to the PN map file.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
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

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # Prepare image.
    rgbimg = cv2.imread(args.img_path)
    pnmaps = tiff_imread(args.pnmap_path)
    pnmaps = np.transpose(pnmaps, [1,2,0])
    img = np.concatenate((rgbimg, pnmaps), axis=2)
    h, w, ch = img.shape

    # Extract mean.
    img = np.float32(img) - IMG_MEAN

    # Create network.
    with tf.device('/gpu:1'):
        input_img = tf.placeholder(tf.float32, shape=[None, h, w, ch], name='input_img')
        net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False)

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output = net.layers['fc1_voc12']
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
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

    # pdb.set_trace()
    # Perform inference.
    preds = sess.run(raw_output_up, feed_dict={input_img:img[np.newaxis, ...]})
    preds = preds.squeeze()

    # msk = decode_labels(preds)
    # im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # cv2.imwrite(args.save_dir+'mask.png', preds)
    image.imsave(args.save_dir+'mask.png', preds) # ,cmap = cm.grey, vmin=0, vmax=255 )

    print('The output file has been saved to {}'.format(args.save_dir + 'mask.png'))


if __name__ == '__main__':
    main()

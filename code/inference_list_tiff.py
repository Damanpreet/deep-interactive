"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
import os
import time

import cv2
import numpy as np
from matplotlib import image # , cm
from tifffile import imread as tiff_imread

import tensorflow as tf

from deeplab_resnet import cfg
from deeplab_resnet import DeepLabResNetModel #, decode_labels, prepare_label

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--save-dir", type=str, default=cfg.OUT_PATH,
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
     output: img_list: list of fname to rgbimages
             sample_list: list of fname to sample pos/neg map packages and its label.
    '''
    img_list, sample_list = [], []
    with open(cfg.VAL_LIST, 'r') as f:
        for line in f:
            img_name, sample_name = line.split()

            img_list.append(img_name)
            sample_list.append(sample_name)

    return img_list, sample_list



def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Prepare image.
    img_list, sample_list = read_image_list()
    ch = 4 if cfg.ONLY_POS else 5

    # Create network.
    with tf.device('/cpu:0'):
        input_imgs = tf.placeholder(tf.float32, shape=[None, None,None, ch], name='input_img')
        net = DeepLabResNetModel({'data': input_imgs}, is_training=False)

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output     = tf.sigmoid(net.layers['fc1_voc12'])
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
    for img_name, sample_name in zip(img_list, sample_list):
        rgbPath   = os.path.join(cfg.RGB_PATH, img_name+cfg.RGB_EXT)
        pnPath    = os.path.join(cfg.PNSAMPLE_PATH, sample_name+cfg.PNSAMPLE_EXT)
        labelPath = os.path.join(cfg.GT_PATH, sample_name+cfg.GT_EXT)

        # Perform inference.
        rgbimg = cv2.imread(rgbPath)
        pnmaps = tiff_imread(pnPath)
        pnmaps = pnmaps[0, ...] if cfg.ONLY_POS else np.transpose(pnmaps, [1,2,0])
        pnmaps = pnmaps[..., np.newaxis] if cfg.ONLY_POS else pnmaps

        start_time = time.time();
        inData     = np.concatenate((rgbimg, pnmaps), axis=2)
        inData     = np.float32(inData)-cfg.IMG_MEAN[0:4] if cfg.ONLY_POS else np.float32(inData)-cfg.IMG_MEAN

        rOutput    = sess.run(raw_output_up, feed_dict={input_imgs:inData[np.newaxis, ...]})
        rOutput    = rOutput.squeeze()
        img        = np.zeros(rgbimg.shape)
        img[:,:,0] = rOutput*255
        img        = rgbimg[..., [2,0,1]]*0.6 + img*0.5
        img[img>255] = 255
        img          = img.astype(np.uint8)

        # image.imsave(args.save_dir+ sample_fname +'.png', preds.squeeze()*200, cmap = cm.gray, vmin=0, vmax=255 )
        image.imsave(args.save_dir+ sample_name +'_color.png', img) #, cmap = cm.gray, vmin=0, vmax=255 )
        duration = time.time()-start_time

        print('The output file {} has been saved to {}. -- time: {:.3f} sec/({:d}, {:d})'.format(sample_name, args.save_dir, duration, img.shape[0], img.shape[1]))


if __name__ == '__main__':
    main()

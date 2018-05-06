# import sys

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian#, softmax_to_unary

from matplotlib import image as mpimg

import pdb

IMG_DIR = '/home/yuanjial/DataSet/PASCAL_aug/JPEGImages/'
ANN_DIR = './output/'

def denseCRF_refine(imgName, annNum):
    # Get arguments
    #imgName = '2007_001526' #sys.argv[1]
    #annNum  = 5 #int(sys.argv[2])

    # read images and probability map
    img = mpimg.imread(IMG_DIR+imgName+'.jpg')

    annImg = []
    for k in range(annNum):
        im = mpimg.imread(ANN_DIR+imgName+'_'+str(k)+'.png')
        annImg.append(im[..., 0])
    annImg = np.array(annImg)

    # get label
    labels = np.argmax(annImg, axis=0)
    proImg = np.max(annImg, axis=0)
    labels = (labels+1)*(proImg > 0.001)+1

    # mpimg.imsave(imgName+'_out.png', labels)

    # dence crf inference
    annImg = annImg/max(np.max(annImg), 1e-10)
    bgImg  = (1 - np.max(annImg, axis=0))*0.05
    U      = np.concatenate((bgImg[np.newaxis, ...], annImg), axis=0)
    U      = -np.log(U+1e-10)
    U      = U.reshape([U.shape[0], U.shape[1]*U.shape[2]])


    n_labels = annNum+1
    d = dcrf.DenseCRF(img.shape[1]*img.shape[0], n_labels)
    #U2 = unary_from_labels(labels, n_labels, gt_prob=0.5, zero_unsure=True)
    #U  = U*0.5 + U2*0.5
    d.setUnaryEnergy(U)

    feats = create_pairwise_gaussian(sdims=(9, 9), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,  kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(40, 40), schan=(17, 17, 17), img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)
    fineLabel = np.argmax(Q, axis=0)
    mpimg.imsave(ANN_DIR+imgName+'_out_fine.png',fineLabel.reshape(labels.shape[0:2]))

    return fineLabel.reshape(labels.shape[0:2]), (labels-1)


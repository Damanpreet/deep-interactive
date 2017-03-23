# import sys

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian #, unary_from_softmax, unary_from_labels
from matplotlib import image as mpimg
from matplotlib import cm as mpcm
import pdb

IMG_DIR = '/home/yuanjial/DataSet/PASCAL_aug/JPEGImages/'
# dense CRF refinement
def denseCRFrefinement(predMap, fname, iterNum):
    # prprocess proMap.
    rgbImg = mpimg.imread(IMG_DIR+fname+'.jpg')
    proMap = []
    for k in range(predMap.shape[0]):
        mpimg.imsave('temporal.png', predMap[k].squeeze(), cmap=mpcm.gray, vmin=0, vmax=1)
        img = mpimg.imread('temporal.png')
        proMap.append(img[...,0])
    proMap = np.array(proMap)

    #pdb.set_trace()
    # compute coarse segmentation label
    labels = np.argmax(proMap, axis = 0)
    proImg = np.max(proMap, axis = 0)
    labels = (labels+1)*(proImg > 0.001)+1

    # construct unary map
    proMap = (proMap)/max(np.max(proMap), 1e-10)
    bgImg  = (1- np.max(proMap, axis=0))*0.1
    uMap   = np.concatenate((bgImg[np.newaxis, ...], proMap), axis=0)
    uMap   = -np.log(uMap + 1e-10)
    # uMap   = softmax_to_unary(uMap)
    uMap   = uMap.reshape([uMap.shape[0], uMap.shape[1]*uMap.shape[2]])

    # dence crf inference
    n_labels = uMap.shape[0]
    d = dcrf.DenseCRF(rgbImg.shape[1]*rgbImg.shape[0], n_labels)
    # uMap = unary_from_labels(labels, n_labels, gt_prob=0.5, zero_unsure=True)
    d.setUnaryEnergy(uMap)

    feats = create_pairwise_gaussian(sdims=(9, 9), shape=rgbImg.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,  kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13), img=rgbImg, chdim=2)
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(iterNum)
    fineLabel = np.argmax(Q, axis=0)
    fineLabel = fineLabel.reshape([rgbImg.shape[0], rgbImg.shape[1]])
    mpimg.imsave(fname+'_fine.png', fineLabel)
    return fineLabel, (labels-1)


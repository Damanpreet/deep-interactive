import os.path as osp
import os
import time
import cv2
import scipy.misc as smisc
# import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.morphology as smorpho
import random as ran
import argparse
from tifffile import imsave as tImsave

from cfg.config import cfg

def load_image_path(img_dir, img_ext):
    '''Load training image text file as list
    @img_dir: image directory name.
    @img_ext: image extension type.
    output: list of image file path
    '''
    images = []
    file_names = []
    if not img_dir:
        print('No image directory provided.')
        return images

    with open(cfg.TXT_PATH, 'r') as fp:
        for line in fp:
            # truncate /n character
            file_names.append(line[:-1])
            file_name = line[:-1] + img_ext
            images.append(osp.join(cfg.BASE_DIR, img_dir, file_name))

    return images, file_names


def euclidean_dis(p1, p2):
    return np.linalg.norm(p1-p2)


def sample_cands(candidates, is_fixed, min_num, max_num, criteria, *args):
    res = []

    if not is_fixed:
        num = ran.choice(range(min_num, max_num))

    # to prevent deadlock when there are no enough points to satisfy the criteria
    cnt = 0

    # Collecting num points
    while len(res) != num and cnt < 100:
        cnt += 1
        cand = ran.choice(candidates)

        # If satisfy criteria, add to result
        if criteria(res, cand, *args):
            res.append(cand)


    return res


def dis_criteria(points, point, least_dist):
    if len(points) == 0:
        return True

    diff = points - point
    dist = np.sqrt(np.sum(diff*diff, 1))

    if(min(dist) < least_dist):
        return False

    return True


def _compute_step(ys, xs):

    min_x, min_y, max_x, max_y = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
    h        = max_y - min_y
    w        = max_x - min_x
    min_dist = min(h, w)

    d_step = (min_dist - cfg.D_MARGIN) // cfg.RATIO_FACTOR

    return d_step

def _erosion_and_remove(valid_area, erode_dim=(2*cfg.D_MARGIN+1, 2*cfg.D_MARGIN+1)):
    erode_kernel = np.ones(erode_dim)
    erode = cv2.erode(valid_area, erode_kernel, iterations=1)
    donut = valid_area - erode
    return donut


def _erosion(valid_area, erode_dim=(2*cfg.D_MARGIN+1, 2*cfg.D_MARGIN+1)):
    # Fixed 3 by 3 erosion
    erode_kernel = np.ones(erode_dim)
    erode = cv2.erode(valid_area, erode_kernel, iterations=1)

    return erode

def _dilation_and_remove(valid_area, dilation_dim=(2*cfg.D+1, 2*cfg.D+1)):
    dilation_kernel = np.ones(dilation_dim)
    dilation = cv2.dilate(valid_area, dilation_kernel, iterations=1)
    dilation -= valid_area
    return dilation

def _dilation_and_remove_and_erosion(valid_area):
    dilation_dim = (2 * cfg.D + 1, 2 * cfg.D + 1)
    dilation     = _dilation_and_remove(valid_area, dilation_dim)

    erode_dim = (2 * cfg.NEG3_MARGIN + 1, 2 * cfg.NEG3_MARGIN + 1)
    erode     = _erosion(dilation, erode_dim)

    return erode


def _filter_samples(img_arr, is_fixed, morphology, min_num=0, max_num=cfg.N_POS):
    '''Sampling points from the pixels that get from morphology function for each object
    based on following rules:
        1. Any two pixels are at least d_step pixels away from each other;
        2. Any pixels is at least d_margin pixels away from object boundaries;

    @img_arr: instance label image
    @morphology: function to get desired sample area
    output: a dictionary of positive points for each object per entry
    '''
    locs = {}

    obj_lst = np.unique(img_arr[img_arr != 0])

    d_step = 1

    obj_mappings = {}
    for obj in obj_lst:
        # numpy logic operation for masking
        obj_mappings[obj] = img_arr * (img_arr == obj)
        locs[obj] = []


    for obj, mapping in obj_mappings.items():

        # Get desired sample area
        morph = morphology(mapping)

        # Prevent the segmentation is too thin to erode.
        if np.sum(morph) == 0:
            morph = np.array(mapping, dtype=np.uint8)

        ys, xs = np.where(morph == obj)
        d_step = _compute_step(ys, xs)


        # Find indexes of where the element equal to obj.
        candidates = np.array(list(zip(*np.where(morph== obj))))


        # Sample points from candidates
        samples = sample_cands(candidates, is_fixed, min_num, max_num, dis_criteria, d_step)
        locs[obj].extend(samples)

        #for s in samples:
        #    morph[s[0], s[1]] = 200
        #print(samples)
        #plt.figure()
        #plt.subplot(221)
        #plt.imshow(morph)
        #plt.subplot(222)
        #plt.imshow(mapping)
        #plt.show()

    return locs


def pos_strategy(img_arr, num_poss=cfg.N_POS):
    '''Positive points sampling strategy.
    @img_arr: instance label image
    @output: a dictionary of sampled negative points
    '''
    return _filter_samples(img_arr, False, _erosion_and_remove, 0, num_poss)


def neg_strategy1(img_arr, num_negs=cfg.N_NEG):
    '''Negative points sampling strategy 1.
    @img_arr: instance label image
    @num_negs: number of negative points needed to be sampled
    output: a dictionary of sampled negative points
    '''
    return _filter_samples(img_arr, False, _dilation_and_remove, (2*num_negs)//3, num_negs)


def neg_strategy2(img_arr, num_negs=cfg.N_NEG):
    '''Negative points sampling strategy 2.
    @img_arr: instance label image
    @num_negs: number of negative points needed to be sampled
    return {
        'a': 1,
        'b': 2,
    }[x]mage path
    output: a dictionary of sampled negative points
    '''
    locs = {}

    obj_lst = np.unique(img_arr[img_arr != 0])


    obj_mappings = {}
    for obj in obj_lst:
        obj_mappings[obj] = img_arr * (img_arr == obj)
        locs[obj] = []

    for obj, mapping in obj_mappings.items():
        for other_obj, other_mapping in obj_mappings.items():
            if obj != other_obj:
                morph = _erosion(other_mapping)

                if np.sum(morph) == 0:
                    morph = np.array(other_mapping, dtype=np.uint8)

                ys, xs = np.where(morph == other_obj)

                d_step = _compute_step(ys, xs)

                candidates = np.array(list(zip(*np.where(morph == other_obj))))

                samples = sample_cands(candidates, False, (2*num_negs)//3, num_negs, dis_criteria, d_step)

                locs[obj].extend(samples)

                #for s in samples:
                #    morph[s[0], s[1]] = 200
                #print(samples)
                #plt.figure()
                #plt.subplot(221)
                #plt.imshow(morph)
                #plt.subplot(222)
                #plt.imshow(mapping)
                #plt.show()

    return locs

def neg_strategy3(img_arr, num_negs=cfg.N_NEG):
    '''Negative points sampling strategy 3.
    @img_arr: instance label image
    @num_negs: number of negative points needed to be sampled
    output: a dictionary of sampled negative points
    '''
    return _filter_samples(img_arr, True, _dilation_and_remove_and_erosion, num_negs)


def cat_channels(gt):
    '''Constructing positive mapping channel and negative mapping channels.
    input:
    output:
    '''

    res = {}
    res['data'] = []
    res['labels'] = []
    if not gt:
        return None

    gt_arr  = smisc.imread(gt, mode='P')
    h, w    = gt_arr.shape

    for n in range(cfg.N_PAIRS):
        pos_samples = pos_strategy(gt_arr)
        neg_samples = neg_strategy1(gt_arr)

        # neg_samples = [neg_strategy1, neg_strategy2, neg_strategy3]
        # idx = n // (cfg.N_PAIRS)
        # neg_samples = neg_samples[idx](gt)

        start_time = time.time()
        pos_energies = construct_channels(pos_samples, h, w)
        neg_energies = construct_channels(neg_samples, h, w)
        duration = time.time()-start_time;
        print('construct pos/neg channel from points: {:.5f}sec / ({:d}, {:d})'.format(duration, h, w))

        for obj in pos_samples:
            pairs = np.concatenate((pos_energies[obj], neg_energies[obj]), axis=2)
            label = gt_arr * (gt_arr == obj)
            res['data'].append(pairs)
            res['labels'].append(label)

    return res


def construct_channels(samples, h, w):

    energies = {}
    for obj, lst in samples.items():
        mask_obj    = np.ones((h, w))
        if(len(lst)==0):
            energy = mask_obj*255
        else:
            for pt in lst:
                mask_obj[pt[0],pt[1]] = 0

            energy = smorpho.distance_transform_edt(mask_obj)
            energy = energy * cfg.ENERGY_SCALE
            energy[energy>255] = 255

        energy        = np.uint8(energy)
        energy        = energy[:, :, np.newaxis]
        energies[obj] = energy

    return energies

def argparser():
    ''' Parse all arguments provided from the command line tool(CLT)
    '''
    parser = argparse.ArgumentParser(description='Convert images to five channels mapping.')
    parser.add_argument('--save-dir', type=str, default='converted', help="Directory to store converted tiff files")
    return parser.parse_args()


def create_dir(dir_name):
    if not osp.exists(dir_name):
        os.makedirs(dir_name)

def main():

    with open(cfg.TXT_PATH) as f:
        image_list = f.read().splitlines()

    gts, filenames = [], []
    for fname in image_list:
        gts.append(osp.join(cfg.INSTANCEANN_DIR, fname+cfg.GT_EXT))
        filenames.append(fname)


    output_path = cfg.OUT_PATH
    create_dir(output_path)

    labels_dir = osp.join(output_path, 'labels')
    create_dir(labels_dir)

    data_dir = osp.join(output_path, 'data')
    create_dir(data_dir)

    train_txt = open(cfg.OUT_TXT_PATH, 'w+')
    train_txt.truncate()

    for idx, (gt, fname) in enumerate(zip(gts, filenames)):
        # import pdb
        # pdb.set_trace()
        if(not os.path.isfile(gt)):
            continue

        pairs = cat_channels(gt)

        data = np.array(pairs['data'], dtype=np.uint8)
        labels = np.array(pairs['labels'], dtype=np.uint8)

        for i in range(data.shape[0]):
            sample = data[i,...]
            label  = labels[i,...]

            sample = sample.transpose((2, 0, 1))

            new_name   = fname + '-' + str(i)
            file_path  = osp.join(data_dir, new_name+'_pnsamples.tif')
            label_path = osp.join(labels_dir, new_name+'_label.tif')

            train_txt.write(fname + ' ' + new_name + '\n')
            tImsave(file_path, sample, compress=6)
            tImsave(label_path, label*255, compress=6)

        print('image {} done!, counter={}'.format(fname, idx))

    train_txt.close()

if __name__ == '__main__':
    main()



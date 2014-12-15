#!/usr/bin/env python
# -*- coding: utf-8; -*-

'''
-----------------------------------------------------------------------------
Image processing utility functions.
-----------------------------------------------------------------------------
'''

import gzip
import pickle
import logging

import numpy as np

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler

from skimage.io import imread
from skimage.util import pad
from skimage.transform import resize, rescale, downscale_local_mean
from skimage.filter import threshold_otsu
from skimage.filter.rank import median
from skimage.measure import label, moments, regionprops
from skimage.morphology import disk, skeletonize, erosion

#-----------------------------------------------------------------------------
log = logging.getLogger('main')

def prepare(img, angle=1.80):
    img = median(img, disk(4))
    return img

def bbox_to_slice(bbox):
    return np.s_[bbox[0]:bbox[2], bbox[1]:bbox[3]]

def largest_object(img):
    labels = label(img, background=False, neighbors=8)
    labels[labels != -1] += 1

    props = [(prop.area, prop) for prop in regionprops(labels)]
    if not props:
        return None

    area, prop = max(props)
    return prop.image

def padtosize(img, h, w):
    ah, aw = img.shape
    h = (h - ah) / 2
    w = (w - aw) / 2
    return pad(img, ((h, h+(ah%2)), (w, w+(aw%2))), mode='constant')

def roi(img):
    # Select regions of interest (first and second digit).
    A = img[110:350, 15:135]
    B = img[120:360, 265:385]
    C = img[125:365, 520:640]

    assert A.shape == B.shape == C.shape
    shape = A.shape

    # Apply Otsu's thresholding to the regions.
    A = A > threshold_otsu(A)
    B = B > threshold_otsu(B)
    C = C > threshold_otsu(C)

    # Morphological transforms.
    r = disk(3)
    A = erosion(A, r) // 255
    B = erosion(B, r) // 255
    C = erosion(C, r) // 255

    # Skeletonize and narrow down to largest region.
    A = largest_object(skeletonize(A))
    B = largest_object(skeletonize(B))
    C = largest_object(skeletonize(C))

    # Pad back into the original size.
    A = padtosize(A, shape[0], shape[1])
    B = padtosize(B, shape[0], shape[1])
    C = padtosize(C, shape[0], shape[1])

    # Rescale.
    A = downscale_local_mean(A, (5,5))
    B = downscale_local_mean(B, (5,5))
    C = downscale_local_mean(C, (5,5))

    return A, B, C

def imgdir_to_array(topdir):
    images, labels, paths = [], [], []

    dirs = (i for i in topdir.iterdir() if i.is_dir())
    for dir in dirs:
        label = int(dir.name[0])
        for path in dir.iterdir():
            img = imread(str(path), 0)
            images.append(img)
            labels.append(label)
            paths.append(path)

    return np.array(images), np.array(labels), paths

def loadmodels(models):
    with gzip.open(models, 'rb') as fh:
        log.info('%s ...', fh.name)
        pca_A, std_A, knn_A, pca_B, std_B, knn_B, pca_C, std_C, knn_C = pickle.load(fh)

    # knn_A.set_params(n_neighbors=5, weights='distance')
    # knn_B.set_params(n_neighbors=5, weights='distance')
    # knn_C.set_params(n_neighbors=5, weights='distance')
    return pca_A, std_A, knn_A, pca_B, std_B, knn_B, pca_C, std_C, knn_C

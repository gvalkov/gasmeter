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

import cv2
import numpy as np

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler

#-----------------------------------------------------------------------------
log = logging.getLogger('main')

def prepare(img, angle=1.80):
    # Black letters on white background.
    img = np.invert(img)

    # # Straighten-up the image.
    # rows, cols = img.shape
    # cor = (cols/2, rows/2)  # center of rotation
    # M = cv2.getRotationMatrix2D(cor, angle, 1)
    # img = cv2.warpAffine(img, M, (cols, rows))

    return img

def roi(img):
    # Select regions of interest (first and second digit).
    A = img[110:350, 15:135]
    B = img[120:360, 265:385]
    C = img[125:365, 520:640]

    # Manually blank out some regions.
    B[220:, 0:] = 255
    B[:, 0:5] = 255
    C[220:, 0:] = 255
    C[:, 0:5] = 255

    # A = cv2.medianBlur(A, 11)
    # B = cv2.medianBlur(B, 11)
    # C = cv2.medianBlur(C, 11)

    A = cv2.GaussianBlur(A, (5, 5), 0)
    B = cv2.GaussianBlur(B, (5, 5), 0)
    C = cv2.GaussianBlur(C, (5, 5), 0)

    # Apply Otsu's thresholding to the regions.
    _, A = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, B = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, C = cv2.threshold(C, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply adaptive thresholding to the regions.
    # A = cv2.adaptiveThreshold(A, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 8)
    # B = cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 8)
    # C = cv2.adaptiveThreshold(C, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 8)

    # Resize
    r = 50.0 / A.shape[1]
    dim = (50, int(A.shape[0] * r))
    A = cv2.resize(A, dim, interpolation = cv2.INTER_LINEAR)
    B = cv2.resize(B, dim, interpolation = cv2.INTER_LINEAR)
    C = cv2.resize(C, dim, interpolation = cv2.INTER_LINEAR)

    return A, B, C

def imgdir_to_array(topdir):
    images, labels = [], []

    dirs = (i for i in topdir.iterdir() if i.is_dir())
    for dir in dirs:
        label = int(dir.name[0])
        for path in dir.iterdir():
            img = cv2.imread(str(path), 0)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def loadmodels(pA, pB, pC):
    with gzip.open(pA, 'rb') as fh:
        log.info('%s ...', pA)
        knn_A = pickle.load(fh)

    with gzip.open(pB, 'rb') as fh:
        log.info('%s ...', pB)
        knn_B = pickle.load(fh)

    with gzip.open(pC, 'rb') as fh:
        log.info('%s ...', pC)
        knn_C = pickle.load(fh)

    knn_A.set_params(n_neighbors=5, weights='distance')
    knn_B.set_params(n_neighbors=5, weights='distance')
    knn_C.set_params(n_neighbors=5, weights='distance')

    return knn_A, knn_B, knn_C

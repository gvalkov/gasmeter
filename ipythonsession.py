# -*- coding: utf-8; -*-

import os, sys, re
from src.imageutils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime as dt, date


imread = lambda x: cv2.imread(x, 0)
show = s = lambda x: plt.imshow(x, cmap='gray', interpolation='bicubic')

img = np.invert(imread('data/raw/2014-12-07T18:01:39.756752.jpg'))
A, B, C = roi(img)

lm = lambda: loadmodels('./data/knn_A.pickle.gz', './data/knn_C.pickle.gz', './data/knn_C.pickle.gz')

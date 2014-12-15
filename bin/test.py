#!/usr/bin/env python
# -*- coding: utf-8; -*-

import os, sys

from pathlib import Path
from collections import defaultdict

import numpy as np
import tabulate

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split

from src.imageutils import imgdir_to_array


#-----------------------------------------------------------------------------
data, labels, paths = imgdir_to_array(Path(sys.argv[1]))
data = data.reshape((data.shape[0], -1))

res = train_test_split(data, labels, range(len(paths)), test_size=0.1)
data, test, data_labels, test_labels, data_paths, test_paths = res

# std = StandardScaler()
# pca = RandomizedPCA(n_components=10)
# pca.fit(data)

# data_pca = pca.transform(data)
# test_pca = pca.transform(test)

# data_pca = std.fit_transform(data_pca)
# test_pca = std.fit_transform(test_pca)

data_pca = data
test_pca = test

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_pca.reshape(data_pca.shape[0], -1), data_labels)

res = knn.predict(test_pca)

print(classification_report(test_labels, res))

# nit = np.unique(res)
# err = confusion_matrix(test_labels, res)
# err = np.insert(err, 0, nit, axis=1)
# print(tabulate.tabulate(err, headers=nit, tablefmt="orgtbl"))

miss = defaultdict(list)
for i in range(test_pca.shape[0]):
    sample = test_pca[i]
    res = knn.predict(sample)
    true = test_labels[i]
    if res != true:
        miss[true].append((res[0], paths[test_paths[i]]))

for key in sorted(miss):
    print('Digit: %s' % key)
    for prediction, path in miss[key]:
        print('%s: %s' % (prediction, path))
    print('-' * 66)

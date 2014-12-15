#!/usr/bin/env python3
# -*- coding: utf-8; -*-


import cv2
import gzip, pickle
import multiprocessing as mp
import numpy as np

from invoke import task, run
from pathlib import Path
from itertools import product

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from skimage.io import imread, imsave, imread_collection

from src.imageutils import prepare, roi, imgdir_to_array


#-----------------------------------------------------------------------------
datadir = Path('./data/')
training_models   = datadir / 'models.pickle.gz'
training_data_npz = datadir / 'training-data.npz'
training_data_dir = datadir / 'training/'
testing_data_dir  = datadir / 'testing/'
labeled_data_dir  = datadir / 'labeled/'
raw_training_data_dir = datadir / 'raw/'

@task
def prepare_labeled(root='training', parallel=False):
    root = {'training': training_data_dir, 'testing': testing_data_dir}[root]

    # Create all destination directories.
    labels = product('ABC', '0123456789')
    for group, label in labels:
        path = root/group/label
        if not path.exists():
            path.mkdir(parents=True)

    ic = imread_collection('%s/*/*/*.jpg' % labeled_data_dir)
    images = ((ic[n], name, root) for n, name in enumerate(ic.files))

    if parallel:
        pool = mp.Pool(7)
        pool.map(prepare_labeled_inner, images)
    else:
        for img in images:
            prepare_labeled_inner(img)

def prepare_labeled_inner(arg):
    img, name, root = arg

    name = Path(name)
    group, label = name.parts[-3:-1]  # e.g. A, 1

    img = prepare(img)
    A, B, C = roi(img)

    img = vars()[group]  # A, B or C
    dest = root / group / label / name.with_suffix('.png').name
    print('Source: %s\nDest: %s' % (name, dest))

    if any(i is None for i in (A,B,C)):
        print('Invalid input ... skipping.')
        return

    imsave(str(dest), img)

@task(aliases=['training-data'])
def training_data():
    print('Loading images from %s.' % training_data_dir)
    images_A, labels_A, paths_A = imgdir_to_array(training_data_dir/'A')
    images_B, labels_B, paths_A = imgdir_to_array(training_data_dir/'B')
    images_C, labels_C, paths_A = imgdir_to_array(training_data_dir/'C')

    def getcounts(arr):
        counts = np.bincount(arr)
        indexes = np.nonzero(counts)[0]
        return zip(indexes, counts[indexes])

    print('A: %s' % getcounts(labels_A))
    print('B: %s' % getcounts(labels_B))
    print('C: %s' % getcounts(labels_C))

    print('Serializing images to %s.' % training_data_npz)
    np.savez_compressed(str(training_data_npz),
                        images_A=images_A, labels_A=labels_A,
                        images_B=images_B, labels_B=labels_B,
                        images_C=images_C, labels_C=labels_C)

@task
def invert_images(dest='training'):
    dest = {'training': training_data_dir, 'testing': testing_data_dir}[dest]

    for path in dest.rglob('**/*.png'):
        print('%s' % path)

        img = imread(str(path), 0)
        img = np.invert(img)
        imsave(str(path), img)

@task
def train_pca_std():
    print('Loading training data %s.' % training_data_npz)
    npz = np.load(str(training_data_npz))
    images_A, labels_A = npz['images_A'], npz['labels_A']
    images_B, labels_B = npz['images_B'], npz['labels_B']
    images_C, labels_C = npz['images_C'], npz['labels_C']

    def trainset(data, labels):
        pca = RandomizedPCA(n_components=10)
        std = StandardScaler()
        data = np.reshape(data, (data.shape[0], -1))
        data = pca.fit_transform(data)
        data = std.fit_transform(data)
        knn = KNeighborsClassifier()
        knn.fit(data, labels)

        return pca, std, knn

    pca_A, std_A, knn_A = trainset(images_A, labels_A)
    pca_B, std_B, knn_B = trainset(images_B, labels_B)
    pca_C, std_C, knn_C = trainset(images_C, labels_C)

    with gzip.open(str(training_models), 'wb') as fh:
        print('Serializing to %s.' % fh.name)
        res = [pca_A, std_A, knn_A, pca_B, std_B, knn_B, pca_C, std_C, knn_C]
        pickle.dump(res, fh)

@task
def train():
    print('Loading training data %s.' % training_data_npz)
    npz = np.load(str(training_data_npz))
    images_A, labels_A = npz['images_A'], npz['labels_A']
    images_B, labels_B = npz['images_B'], npz['labels_B']
    images_C, labels_C = npz['images_C'], npz['labels_C']

    def trainset(data, labels):
        knn = KNeighborsClassifier()
        knn.fit(data.reshape(data.shape[0], -1), labels)
        return knn

    knn_A = trainset(images_A, labels_A)
    knn_B = trainset(images_B, labels_B)
    knn_C = trainset(images_C, labels_C)

    with gzip.open(str(training_models), 'wb') as fh:
        print('Serializing to %s.' % fh.name)
        res = [knn_A, knn_B, knn_C]
        pickle.dump(res, fh)

@task
def clean():
    paths = [training_models, training_data_npz]
    for path in paths:
        if path.exists():
            print('Removing %s' % path)
            path.unlink()

@task
def deploy():
    run('rsync -vr ./src  ./tasks.py gasmeter.service pi@192.168.0.20:/home/pi/meter/')
    run('rsync -vr ./data/training-data.npz  pi@192.168.0.20:/home/pi/meter/data/')

@task
def pullrawimg():
    run('rsync -vr pi@192.168.0.20:/home/pi/meter/data/raw ./data/')

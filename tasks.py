#!/usr/bin/env python3
# -*- coding: utf-8; -*-


import cv2
import gzip, pickle
import numpy as np
from invoke import task, run
from pathlib import Path

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from src.imageutils import prepare, roi, imgdir_to_array


#-----------------------------------------------------------------------------
datadir = Path('./data/')
training_model_base = datadir / 'knn_%s.pickle.gz'
training_data_npz   = datadir / 'training-data.npz'
training_data_dir   = datadir / 'training/'
raw_training_data_dir = datadir / 'raw/'


@task
def prepare_raw_images():
    if not training_data_dir.exists():
        (training_data_dir/'A').mkdir(parents=True)
        (training_data_dir/'B').mkdir(parents=True)
        (training_data_dir/'C').mkdir(parents=True)

    for path in raw_training_data_dir.iterdir():
        img = cv2.imread(str(path), 0)
        img = prepare(img)
        A, B, C = roi(img)

        dest_A = training_data_dir / 'A' / path.name
        dest_B = training_data_dir / 'B' / path.name
        dest_C = training_data_dir / 'C' / path.name

        print('%s:' % path)
        print('A: %s' % dest_A)
        print('B: %s' % dest_B)
        print('C: %s' % dest_C)

        cv2.imwrite(str(dest_A), A)
        cv2.imwrite(str(dest_B), B)
        cv2.imwrite(str(dest_C), C)

@task(aliases=['training-data'])
def training_data():
    print('Loading images from %s.' % training_data_dir)
    images_A, labels_A = imgdir_to_array(training_data_dir/'A')
    images_B, labels_B = imgdir_to_array(training_data_dir/'B')
    images_C, labels_C = imgdir_to_array(training_data_dir/'C')

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
def train():
    print('Loading training data %s.' % training_data_npz)
    npz = np.load(str(training_data_npz))
    images_A, labels_A = npz['images_A'], npz['labels_A']
    images_B, labels_B = npz['images_B'], npz['labels_B']
    images_C, labels_C = npz['images_C'], npz['labels_C']

    print('Reshaping and packing bits.')
    images_A = np.reshape(images_A, (images_A.shape[0], -1))
    images_B = np.reshape(images_B, (images_B.shape[0], -1))
    images_C = np.reshape(images_C, (images_C.shape[0], -1))

    pca = RandomizedPCA(n_components=10)
    std = StandardScaler()

    images_A = pca.fit_transform(images_A)
    images_B = pca.fit_transform(images_B)
    images_C = pca.fit_transform(images_C)

    images_A = std.fit_transform(images_A)
    images_B = std.fit_transform(images_B)
    images_C = std.fit_transform(images_C)

    knn_A = KNeighborsClassifier()
    knn_B = KNeighborsClassifier()
    knn_C = KNeighborsClassifier()
    knn_A.fit(images_A, labels_A)
    knn_B.fit(images_B, labels_B)
    knn_C.fit(images_C, labels_C)

    with gzip.open(str(training_model_base) % 'A', 'wb') as fh:
        print('Serializing to %s.' % fh.name)
        pickle.dump(knn_A, fh)

    with gzip.open(str(training_model_base) % 'B', 'wb') as fh:
        print('Serializing to %s.' % fh.name)
        pickle.dump(knn_B, fh)

    with gzip.open(str(training_model_base) % 'C', 'wb') as fh:
        print('Serializing to %s.' % fh.name)
        pickle.dump(knn_C, fh)

@task
def clean():
    knn_A = Path(str(training_model_base) % 'A')
    knn_B = Path(str(training_model_base) % 'B')

    paths = [knn_A, knn_B, training_data_npz]
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

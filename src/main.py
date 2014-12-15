#!/usr/bin/env python
# -*- coding: utf-8; -*-

try:
    import cPickle as pickle
except ImportError:
    import pickle

import time
import gzip
import atexit
import logging
import argparse

from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import cv2
import psycopg2
import numpy as np

from . imageutils import roi, prepare, loadmodels
from . capture import Webcam

#-----------------------------------------------------------------------------
log = logging.getLogger('main')
fmt = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=fmt, level=logging.DEBUG)

#-----------------------------------------------------------------------------
class WrongDataError(Exception):
    pass

#-----------------------------------------------------------------------------
def parseopts():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-i', '--interval', default=60.0, type=float)
    arg('-v', '--video-device', default=0)
    arg('-M', '--model', default='data/models.pickle.gz')
    arg('-r', '--saveraw', action='store_true')
    arg('-n', '--dryrun', action='store_true')
    arg('--log-unreliable', action='store_true')
    arg('--dburi',  default='postgresql://rpi@192.168.0.37:5432/homemetrics')
    return parser.parse_args()

def writecm3(cursor, ts, cm3delta, dryrun=False):
    sql = cursor.mogrify('INSERT INTO public."Gas" VALUES (%s, %s)', (ts, cm3delta))
    log.info(sql)
    if not dryrun:
        cursor.execute('INSERT INTO public."Gas" VALUES (%s, %s)', (ts, cm3delta))

def replay_backlog(cursor, backlog, dryrun=False):
    if backlog:
        log.info('Replaying %s values from backlog.', len(backlog))
    for ts in backlog:
        writecm3(cursor, ts, backlog[ts], dryrun)
        del backlog[ts]

def main():
    opts = parseopts()

    #--------------------------------------------------------------------------
    log.info('Connecting to %s.', opts.dburi)
    conn = psycopg2.connect(opts.dburi)
    cursor = conn.cursor()

    backlog = OrderedDict()
    replay_backlog(cursor, backlog, opts.dryrun)

    log.info('Loading pickled models:')
    res = loadmodels(opts.model)
    pca_A, std_A, knn_A, pca_B, std_B, knn_B, pca_C, std_C, knn_C = res

    log.info('Setting up image source.')
    capture = Webcam(0, 7)
    A, B, C = None, None, None

    #--------------------------------------------------------------------------
    def cleanup():
        log.info('Terminating ...')
        capture.webcam.release()
        capture.GPIO.cleanup()
        cursor.close()
        conn.close()
    atexit.register(cleanup)

    #--------------------------------------------------------------------------
    while True:
        try:
            log.info('Capturing image.')
            now = datetime.now()
            img = capture()

            if opts.saveraw:
                dest = Path('./data/raw') / '{}.jpg'.format(now.isoformat(), img)
                log.info('Saving raw image to: %s', dest)
                cv2.imwrite(str(dest), img)
                time.sleep(opts.interval)
                continue

            img = prepare(img)
            img_A, img_B, img_C = roi(img)

            ts, A, B, C = readAB(
                now,
                img_A, img_B, img_C,
                pca_A, pca_B, pca_C,
                std_A, std_B, std_C,
                knn_A, knn_B, knn_C, A, B, C, opts
            )
            # replay_backlog(cursor, backlog, opts.dryrun)
            # if delta is not None:
            #     writecm3(cursor, ts, delta, opts.dryrun)
            # conn.commit()
        except psycopg2.Error as error:
            pass
            # log.exception(error)
            # log.info('Adding %s:%s to backlog', ts, delta)
            # backlog[ts] = delta
            # try:
            #     log.info('Reconnecting to %s.', opts.dburi)
            #     conn = psycopg2.connect(opts.dburi)
            #     cursor = conn.cursor()
            # except:
            #     pass
        except WrongDataError as error:
            log.warn(error)

        log.info('Sleeping for %s seconds.', opts.interval)
        time.sleep(opts.interval)

def readAB(now, img_A, img_B, img_C,
                pca_A, pca_B, pca_C,
                std_A, std_B, std_C,
                knn_A, knn_B, knn_C, old_A, old_B, old_C, opts):
    # Sometimes the camera returns fully black or white images.
    if any(np.unique(i).shape == (1,) for i in (img_A, img_B, img_C)):
        raise WrongDataError('Camera did not return anything meaningful. Skipping.')

    # A_prep = std_A.transform(pca_A.transform(np.reshape(img_A, -1)))
    B_prep = std_B.transform(pca_B.transform(np.reshape(img_B, -1)))
    C_prep = std_C.transform(pca_C.transform(np.reshape(img_C, -1)))

    # new_A = knn_A.predict(A_prep)[0]
    new_B = knn_B.predict(B_prep)[0]
    new_C = knn_C.predict(C_prep)[0]

    # proba_A = knn_A.predict_proba(A_prep)[0][new_A]
    proba_B = knn_B.predict_proba(B_prep)[0][new_B]
    proba_C = knn_C.predict_proba(C_prep)[0][new_C]

    # log.info('A: prediction %s with certainty %s', new_A, proba_A)
    log.info('B: prediction %s with certainty %s', new_B, proba_B)
    log.info('C: prediction %s with certainty %s', new_C, proba_C)

    # if opts.log_unreliable and proba_A <= 0.90:
    #     dest_A = Path('./data/lowproba') / 'A' / '{}-{}-p{}.jpg'.format(new_A, now.isoformat(), proba_A)
    #     log.info('Wrote: %s', dest_A)
    #     cv2.imwrite(str(dest_A), img_A)

    if opts.log_unreliable and proba_B <= 0.90:
        dest_B = Path('./data/lowproba') / 'B' / '{}-{}-p{}.jpg'.format(new_B, now.isoformat(), proba_B)
        log.info('Wrote: %s', dest_B)
        cv2.imwrite(str(dest_B), img_B)

    if opts.log_unreliable and proba_C <= 0.90:
        dest_C = Path('./data/lowproba') / 'C' / '{}-{}-p{}.jpg'.format(new_C, now.isoformat(), proba_C)
        log.info('Wrote: %s', dest_C)
        cv2.imwrite(str(dest_C), img_C)

    # if proba_A <= 0.90 or proba_B <= 0.90 or proba_C <= 0.90:
    if proba_B <= 0.90 or proba_C <= 0.90:
        raise WrongDataError('Probability too low. Skipping.')

    # if old_A == new_A and new_B < old_B:
    #     msg = 'New reading %s%s lower than old reading %s%s.' % (new_A, new_B, old_A, old_B)
    #     raise WrongDataError(msg)
k
    # if old_A and old_B:
    #     delta = ((new_A - old_A) % 10) * 10 + ((new_B - old_B) % 10)
    #     delta = int(delta)
    #     log.info('Delta: %s', delta)
    # else:
    #     delta = None

    return now, None, new_B, new_C


if __name__ == '__main__':
    main()

__author__ = 'miquel'
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


def fastnn(queryDescriptors, trainDescriptors, k):
    """ FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    # flann.knnMatch: (queryDescriptors, trainDescriptors, k[, mask[, compactResult]])
    q= queryDescriptors.astype(np.float32)[np.newaxis]
    t= trainDescriptors.astype(np.float32)
    matches = flann.knnMatch(q,t,k)
    for m in matches[0]:
        print m
    """
    q= queryDescriptors.astype(np.float32)[np.newaxis]
    t= trainDescriptors.astype(np.float32)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree',metric='cosine').fit(t)
    distances, indices = nbrs.kneighbors(q)
    return indices[0]
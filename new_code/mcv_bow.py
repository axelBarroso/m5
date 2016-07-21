# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:35:00 2016

@author: ramon
"""
import pickle                    # save/load data files
import multiprocessing as mp
import numpy as np
import cv2
import scipy.cluster.vq as vq    # for k-means
import sys 

import mcv_tools as tl

def bow_encoding(features, codebook):

    soft_assignment = False
    hist = np.ones((1, codebook.shape[0]))

    if soft_assignment:
        for i in range(len(features)):
            for j in range(len(codebook)):
                d = np.sqrt(((features[i]-codebook[j])**2).sum())
                hist[0, j] += 1/(d+sys.float_info.epsilon)
    else:
        code, distance = vq.vq(features, codebook)   # nearest-neighbour
        for i in range(len(code)):
            c = code[i]
            hist[0, c-1] += 1

    histNorm = hist/sum(hist)    # normalize
    #print sum(hist)
    #print len(code)
    return histNorm    

def bowVector(img,centroids,params, meanX, stdX, pca, fileout=None):
    if isinstance(img, str):
        img = cv2.imread(img)
        
    feats,pos = tl.getFeatImgSample(img,params,0,False,False)
    if pca is not None:
        feats = pca.fit_transform(feats)        
    feats = tl.normalizeFeat(feats,meanX,stdX)[0]
    return bowEncodingSingleImage(feats,centroids, params, fileout, meanX,stdX)

def bowEncodingSingleImage(feats,centroids,params,imfile=None,mean=None, std=None):
    v = bow_encoding(feats, centroids)
    if imfile is None:
        return v
    else:
        f = open(imfile+'.pkl', 'wb')
        pickle.dump(v,f)
        f.close()
    return
    
def computeBoW(TRAINFILES,params,runmultiprocess,bowDir,showFig='False'):
    #
    # 1. join descriptors, to compute vocabulary
    #
    X,idxs = tl.getFeatList(TRAINFILES,params,runmultiprocess,showFig)

    #
    # 2. normalize descriptors
    #
    pca = tl.getPCA(X,params)
    if pca is not None:
        X = pca.fit_transform(X)
    X,meanX,stdX = tl.normalizeFeat(X)
    #
    # 3. compute k-means vocabulary over library data
    #
    print 'compute vocabulary over X'
    centroids,variance = vq.kmeans(X,params['K'],1)
    f = open(bowDir+'BOW_CODING_'+params['descriptor'][0]+'.pkl', 'wb')
    pickle.dump(centroids,f)
    pickle.dump(meanX,f)
    pickle.dump(stdX,f)
    pickle.dump(pca,f)
    f.close()

    # re-load
#    f = open(bowDir+'BOW_CODING_'+params['descriptor'][0]+'.pkl', 'rb')
#    centroids = pickle.load(f)
#    meanX = pickle.load(f)
#    stdX = pickle.load(f)
#    pca = pickle.load(f)
#    f.close()

    #
    # 4. encode library images
    #
    if runmultiprocess:
        pool = mp.Pool(mp.cpu_count()- runmultiprocess if mp.cpu_count()>1 else 0)
        PoolResults = []

    n_im = -1
    for imfile in TRAINFILES:
        n_im += 1
        print 'BoW vector: '+imfile

        imfile2 = bowDir+'bw_'+imfile.split('/')[-1][0:-5]
#        feats = X[idxs[n_im,0]:idxs[n_im,0]+idxs[n_im,1]]
        if runmultiprocess:         ########### start change
#            PoolResults.append(pool.apply_async(bowEncodingSingleImage, args=(feats,centroids,params,imfile2)))
            PoolResults.append(pool.apply_async(bowVector, args=(imfile,centroids,params, meanX, stdX, pca, imfile2)))
        else:
#            bowEncodingSingleImage(feats,centroids,params,imfile2)
            bowVector(imfile,centroids,params, meanX, stdX, pca, imfile2)

    if runmultiprocess:
        pool.close()
        pool.join()

    print 'BoW encoding finished!'



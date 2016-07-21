""" 
    
    FISHER coding
    
    Master in Computer Vision - Barcelona
    
    Author: Francesco Ciompi
         
"""

from sklearn import mixture
import pickle                    # save/load data files
from numpy import *
import cv2
import platform
#if not platform.node().lower().startswith('compute'):
#    from pylab import *       # matplotlib, for graphical plots
import mcv_tools as tl
import multiprocessing as mp
import warnings
from sklearn import preprocessing
import numpy as np
warnings.filterwarnings("ignore")


""" 
    Implementation based on the papers:

    [1] "The devil is in the details: an evaluation of recent feature encoding methods"
    Ken Chatfield, Victor Lempitsky, Andrea Vedaldi, Andrew Zisserman, 2011
    
    [2] "Large-Scale Image Retrieval with Compressed Fisher Vectors"
    Florent Perronnin, Yan Liu, Jorge Sanchez and Herve Poirier, 2010    

"""

#
#    Gaussian Mizture Model (GMM) estimation of data X as a mixture of K pdf
#
def GMM_estimate(X,K):

    clf = mixture.GMM(n_components=K,covariance_type='diag',n_iter=5,thresh=0.1,min_covar=0.01)

    print 'GMM fitting with '+str(K)+' pdfs on '+str(shape(X)[1])+' dimensions'
    clf.fit(X)
    
    # assign values
    pi_k = clf.weights_
    mu_k = clf.means_
    sigma_k = clf.covars_
    
    print 'GMM fitting finished!'
    
    return clf, pi_k, mu_k, sigma_k
    
    

#
#    Fisher encoding
# 
def fisher_encoding(X,gmm):

    # the input is one image!
    
    N = shape(X)[0]
    K = gmm.means_.shape[0]
    D = gmm.means_.shape[1]
        
    pi_k = gmm.weights_
    mu_k = gmm.means_
    SIGMA_k = gmm.covars_
        
    # Eq. (2) in [1]
    logprob, P = gmm.score_samples(X)
    Pi_k = tile(pi_k,(N,1))    
    den = reshape(sum(P * Pi_k,axis=1),(N,1))    
    qki = ((P * Pi_k)/tile(den,(1,K))).T
        
    # Eq. (3) in [1]
    F = zeros((1,2*D*K))
    pos = 0
    for k in range(K):
        u_k = 1/(N*sqrt(pi_k[k])) * sum(qki[k] *  dot(sqrt(linalg.inv(diag(SIGMA_k[0]))),(X - tile(mu_k[k],(N,1))).T),axis=1)
        v_k = 1/(N*sqrt(2*pi_k[k])) * sum(qki[k] * ((X - tile(mu_k[k],(N,1))).T * dot(linalg.inv(diag(SIGMA_k[0])),(X - tile(mu_k[k],(N,1))).T)-1),axis=1)
                
        F[0,pos*D:(pos+1)*D] = u_k; pos+=1
        F[0,pos*D:(pos+1)*D] = v_k; pos+=1
    
    return F        
    

clf = []
meanX = []
stdX = []

def FisherVector(img, params, clf, meanX, stdX, pca, fileout=None):
    if isinstance(img, str):
        img = cv2.imread(img)

    feats,pos = tl.getFeatImgSample(img,params,0,False,False)
    if pca is not None:
        feats = pca.fit_transform(feats)    
    feats = tl.normalizeFeat(feats,meanX,stdX)[0]
    return FisherEncodingSingleImage(feats, False, imfile=fileout, model=(clf,meanX,stdX))

def FisherEncodingSingleImage(feats,showFig,imfile=None,model=None):

    global clf, meanX, stdX
    if not (model is None):
        clf = model[0]
        meanX = model[1]
        stdX = model[2]
    else:
        pass
    # 'normal' fisher coding
          
    F = fisher_encoding(feats,clf)
    if imfile is None:
        return F
    else:
        f = open(imfile+'.pkl', 'wb')
        pickle.dump(F,f)
        f.close()
       
            
#
#    Fisher vector of an entire dataset
#
def computeFisherVectors(TRAINFILES,params,runmultiprocess,fisherDir,showFig='False'):
    
    global clf, meanX, stdX

    #    
    # 1. join descriptors, to compute GMM
    #

    X,idxs = tl.getFeatList(TRAINFILES,params,runmultiprocess,showFig)

    print 'Size data: [ ' + str(shape(X)[0]) + ' , ' + str(shape(X)[1]) + ' ]'

    #    
    # 2. normalize descriptors
    #    
    pca = tl.getPCA(X,params)
    if pca is not None:
        X = pca.fit_transform(X)
    X,meanX,stdX = tl.normalizeFeat(X)

    #
    # 3. compute Gaussian Mixture Models over library data
    #    
    print 'compute GMM over X'
    clf, pi_k, mu_k, sigma_k = GMM_estimate(X,params['K'])

    # 4. Normalization
    print 'Normalization of the GMM vectors'
    norm = np.vstack((mu_k,sigma_k))
    norm = preprocessing.normalize(norm)
    clf.means_ = norm[0:params['K']]
    clf.covars_ = norm[params['K']:]

    f = open(fisherDir+'FISHER_GMM_'+params['descriptor'][0]+'.pkl', 'wb')
    pickle.dump(clf,f)
    pickle.dump(meanX,f)
    pickle.dump(stdX,f)
    pickle.dump(pca,f)
    f.close()

    #
    # 4. encode library images
    #
    if runmultiprocess: 
        pool = mp.Pool(mp.cpu_count()- runmultiprocess if mp.cpu_count()>1 else 0)
        PoolResults = []

    n_im = -1
    for imfile in TRAINFILES:
        n_im += 1
        print 'fisher vector: '+imfile
               
        imfile2 = fisherDir+'fv_'+imfile.split('/')[-1][0:-5]
#        feats = X[idxs[n_im,0]:idxs[n_im,0]+idxs[n_im,1]]
        
        if runmultiprocess:         ########### start change
#            PoolResults.append(pool.apply_async(FisherEncodingSingleImage, args=(feats,False,imfile2)))    
            PoolResults.append(pool.apply_async(FisherVector, args=(imfile, params, clf, meanX, stdX, pca, imfile2)))    
        else:
#            FisherEncodingSingleImage(feats,showFig,imfile2)
            FisherVector(imfile, params, clf, meanX, stdX, pca, imfile2)


    if runmultiprocess:
        pool.close()
        pool.join()
            
    
    print 'Fisher encoding finished!'

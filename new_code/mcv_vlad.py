import pickle                    # save/load data files
from numpy import *
import platform
#if not platform.node().lower().startswith('compute'):
#    from pylab import *       # matplotlib, for graphical plots
import multiprocessing as mp
import scipy.cluster.vq as vq
import cv2
import mcv_tools as tl
import sklearn.preprocessing as preprocessing

#warnings.filterwarnings("ignore")


def vlad_encoding(features, codebook):
    (codes, dist) = vq.vq(features, codebook)
    vlad = zeros(codebook.shape)
    index=0
    for idx in codes:
        diff = subtract(features[index], codebook[idx])
        vlad[idx] = add(diff, vlad[idx])
        index+=1

    #Row normalization: L2 normalize vector for each centroid (Added)
    for r in range(vlad.shape[1]):
        vlad[r] = preprocessing.normalize(vlad[r], norm='l2')

    # Column normalization: L2 normalize of all vectors
    vlad = preprocessing.normalize(vlad, norm='l2')

    out = vlad.ravel('F')
    return out

def vladVector(img,centroids,params, meanX, stdX, pca, fileout=None):
    if isinstance(img, str):
        img = cv2.imread(img)

    feats,pos = tl.getFeatImgSample(img,params,0,False,False)
    if pca is not None:
        feats = pca.fit_transform(feats)
    feats = tl.normalizeFeat(feats,meanX,stdX)[0]
    return vladEncodingSingleImage(feats,centroids, params, fileout, meanX,stdX)

def vladEncodingSingleImage(feats,centroids,params,imfile=None,mean=None, std=None):
    #vlad_encoding substracts the mean and assigns the vlad vector
    v = vlad_encoding(feats, centroids)

    if imfile is None:
        return v
    else:
        f = open(imfile+'.pkl', 'wb')
        pickle.dump(v,f)
        f.close()
    return
    
def computeVLAD(TRAINFILES,params,runmultiprocess,vladDir,showFig='False'):
    #
    # 1. join descriptors, to compute vocabulary
    #
    X,idxs = tl.getFeatList(TRAINFILES,params,runmultiprocess, showFig )

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
    f = open(vladDir+'VLAD_CODING_'+params['descriptor'][0]+'.pkl', 'wb');
    pickle.dump(centroids,f)
    pickle.dump(meanX,f)
    pickle.dump(stdX,f)
    pickle.dump(pca,f)
    f.close()

    # re-load
#    f = open(vladDir+'VLAD_CODING_'+params['descriptor'][0]+'.pkl', 'rb')
#    centroids = pickle.load(f)
#    meanX = pickle.load(f)
#    stdX = pickle.load(f)
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
        print 'VLAD vector: '+imfile

        imfile2 = vladDir+'vl_'+imfile.split('/')[-1][0:-5]
#        feats = X[idxs[n_im,0]:idxs[n_im,0]+idxs[n_im,1]]
        if runmultiprocess:         ########### start change
#            PoolResults.append(pool.apply_async(vladEncodingSingleImage, args=(feats,centroids,params,imfile2)))
            PoolResults.append(pool.apply_async(vladVector, args=(imfile,centroids,params, meanX, stdX, pca, imfile2)))
        else:
#            vladEncodingSingleImage(feats,centroids,params,imfile2)
            vladVector(imfile,centroids,params, meanX, stdX, pca, imfile2)

    if runmultiprocess:
        pool.close()
        pool.join()

    print 'VLAD encoding finished!'


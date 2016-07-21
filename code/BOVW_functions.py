import sys
import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import time
import random
import skimage.feature as feat
import scipy.cluster.vq as vq
from sklearn.cluster import KMeans
from aux_functions import *
from sklearn import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from skimage.util import pad
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics.pairwise import additive_chi2_kernel

# *************************
# parameters
# *************************
parameters = np.array([[[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   0.00000000e+00,   0.00000000e+00],\
                     [ -3.90823298e-02,   3.85324254e-02,   2.36880030e-01,   4.62878817e-01,   0.00000000e+00,   0.00000000e+00],\
                     [ -9.86970116e-01,  -8.51969023e-01,  -7.94921771e-01,  -9.95040273e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  9.00991049e-01,   5.24362727e-01,   9.99376086e-01,   9.39107161e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  1.72499209e+00,   5.00000000e+00,   5.65279390e-01,   7.54842803e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,  -0.00000000e+00,  -0.00000000e+00],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   0.00000000e+00,   0.00000000e+00]],\
                    [[  0.00000000e+00,   0.00000000e+00,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   7.75874556e-01,   5.75756054e-01,   4.49437961e-01,   4.49316962e-01],\
                     [  0.00000000e+00,   0.00000000e+00,  -5.02014629e-01,  -1.72492737e-01,  -2.76553982e-01,  -3.06421502e-01],\
                     [  0.00000000e+00,   0.00000000e+00,   5.65279390e-01,   7.54842803e-01,   1.99511862e+00,   1.02801414e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   5.17227023e-01,   4.78353741e-01,   8.39910145e-01,   7.90983657e-01],\
                     [ -0.00000000e+00,  -0.00000000e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  0.00000000e+00,   0.00000000e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  5.83826210e-01,   7.18827304e-01,   1.06878170e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  2.54038810e-01,   1.19930098e-01,   1.16144803e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  1.72499209e+00,   5.00000000e+00,   5.17227023e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  8.44028374e-01,   6.92404886e-01,   8.44786041e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,  -0.00000000e+00,  -0.00000000e+00,  -0.00000000e+00],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00]],\
                    [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.39830359e+00,   1.29424234e+00,   1.26437482e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   9.90182597e-02,   2.14168287e-01,   2.83461782e-01],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.78353741e-01,   8.39910145e-01,   7.90983657e-01],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   7.28741639e-01,   8.56004138e-01,   9.57414544e-01],\
                     [ -0.00000000e+00,  -0.00000000e+00,  -0.00000000e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  1.82483514e+00,   1.69072643e+00,   1.68694113e+00,   1.66981459e+00,   1.78496461e+00,   1.85425811e+00],\
                     [  2.34898971e+00,   2.10250398e+00,   1.90897812e+00,   1.89047978e+00,   1.72030985e+00,   1.74612759e+00],\
                     [  8.44028374e-01,   6.92404886e-01,   8.44786041e-01,   7.28741639e-01,   8.56004138e-01,   9.57414544e-01],\
                     [  1.95218383e+00,   9.58110786e-01,   5.99760697e-01,   6.43832855e-01,   7.36644526e-01,   9.00009936e-01],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  3.91978603e+00,   3.67330031e+00,   3.47977445e+00,   3.46127611e+00,   3.29110617e+00,   3.31692392e+00],\
                     [ -2.56831073e+00,  -2.59153227e+00,  -2.58725114e+00,  -2.59660745e+00,  -2.63248600e+00,  -2.60808914e+00],\
                     [  1.95218383e+00,   9.58110786e-01,   5.99760697e-01,   6.43832855e-01,   7.36644526e-01,   9.00009936e-01],\
                     [  1.01425447e+00,   9.16607781e-01,   8.00824361e-01,   7.55213599e-01,   4.74809095e-01,   5.98984911e-01],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [ -9.97514408e-01,  -1.02073594e+00,  -1.01645481e+00,  -1.02581112e+00,  -1.06168967e+00,  -1.03729281e+00],\
                     [ -1.60987866e+00,  -1.84515592e+00,  -1.96583055e+00,  -2.16285442e+00,  -2.13886579e+00,  -2.13569227e+00],\
                     [  1.01425447e+00,   9.16607781e-01,   8.00824361e-01,   7.55213599e-01,   4.74809095e-01,   5.98984911e-01],\
                     [  9.00991049e-01,   1.10031784e+00,   6.23673475e-01,   5.00000000e+00,   1.73558321e+00,   1.93131386e+00],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  0.00000000e+00,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  0.00000000e+00,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  0.00000000e+00,  -2.74359589e-01,  -3.95034220e-01,  -5.92058089e-01,  -5.68069463e-01,  -5.64895945e-01],\
                     [  0.00000000e+00,  -1.53226390e+00,  -1.33391630e+00,  -1.10791751e+00,  -1.12135837e+00,  -1.12147936e+00],\
                     [  0.00000000e+00,   1.10031784e+00,   6.23673475e-01,   5.00000000e+00,   1.73558321e+00,   1.93131386e+00],\
                     [  0.00000000e+00,   5.24362727e-01,   9.99376086e-01,   9.39107161e-01,   1.99511862e+00,   1.02801414e+00],\
                     [ -0.00000000e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  0.00000000e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  0.00000000e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  0.00000000e+00,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]]])

paramsAchro = np.array([[ 28.28252201,  -0.71423449],\
                    [ 28.28252201,   0.71423449],\
                     [ 79.64930057,  -0.30674052],\
                     [ 79.64930057,   0.30674052]])

thrL = np.array([  0,  31,  42,  51,  66,  76, 150], dtype=np.uint8)

def prepareFiles(rootpath):
    current_GT_id = 0
    filenames = []
    GT_ids = []
    GT_labels = []
    classpath = sorted(glob.glob(rootpath + '*'))
    for i in classpath:
        filespath = sorted(glob.glob(i + '/*.jpg'))
        for j in filespath:
            filenames.append(j)
            GT_ids.append(current_GT_id)
            GT_labels.append(i.split('/')[-1])
        current_GT_id += 1
    return (filenames, GT_ids, GT_labels)

def performPCA(descriptors_train, descriptors_test, isRandomized):

    train_PCA = []
    test_PCA = []

    start=time.time()

    train = []
    for i in range(len(descriptors_train)):
        for keyPoint in range(descriptors_train[i].shape[0]):
            train.append(descriptors_train[i][keyPoint])

    print 'len train matrix: ' +str(train.__len__())

    endPrepareData = time.time()

    print 'Total time Prepare Data: ' +str(endPrepareData-start)

    if isRandomized:
        pca = RandomizedPCA(n_components=0.9)
    else:
        pca = PCA(n_components=0.9)

    pca.fit(train)

    endFitData = time.time()

    print 'Total time Fit Data: ' +str(endFitData-endPrepareData)+ ' secs.'

    for img in range(len(descriptors_train)):
        train_PCA.append(pca.transform(descriptors_train[img]))

    # train_PCA = pca.transform(descriptors_train)

    for img in range(len(descriptors_test)):
        test_PCA.append(pca.transform(descriptors_test[img]))

    # test_PCA = pca.transform(descriptors_test)

    end = time.time()

    print 'Total time perform PCA to test and train: ' +str(end-endFitData)+ ' secs.'
    print 'Total time: ' +str(end-start)+ ' secs.'
    return train_PCA, test_PCA


def data_whitening(descriptorsTrain, descriptorsTest):
    #Obtain the dimensions
    desTrain = descriptorsTrain[0]
    desTest = descriptorsTest[0]

    #fil = len(des)
    colTrain = len(desTrain[1, :])
    colTest = len(desTest[1, :])

    #img_num = len(descriptors)
    D_train = []
    D_test = []

    #compute mean value
    mean_aux = np.zeros([1, colTrain])
    cont = 0
    for img in descriptorsTrain:
        for i in range(0, colTrain):
            mean_aux[0, i] = mean_aux[0, i] + np.sum(img[:, i])
        cont = cont + len(img)

    mean = mean_aux / cont

    #compute standard deviation
    std_aux = np.zeros([1, colTrain])
    for img in descriptorsTrain:
        for i in range(0, colTrain):
            std_aux[0, i] = std_aux[0, i] + np.sum((img[:, i] - mean[0, i])**2)

    std = np.sqrt(std_aux)

    #compute data whitening
    for img in descriptorsTrain:
        img_whiten = np.zeros([len(img),len(img[1,:])])
        for i in range(0, colTrain):
            img_whiten[:, i] = (img[:, i] - mean[0, i]) / std[0, i]
        D_train.append(img_whiten)

    for img in descriptorsTest:
        img_whiten = np.zeros([len(img),len(img[1,:])])
        for i in range(0, colTest):
            img_whiten[:, i] = (img[:, i] - mean[0, i]) / std[0, i]
        D_test.append(img_whiten)

    return D_train, D_test

def getScoreKeyPoint(keyPoint):
    return keyPoint.response


def getKeypointsDescriptors(filenames,detector_type,descriptor_type, color_constancy, GT_ids, GT_labels, num_samples):

    if detector_type is "SURF":
        thresh = 500
        highQuality = 1
        detector = cv2.SURF(thresh, 4, 2, highQuality, 1)
    else:
        detector = cv2.FeatureDetector_create(detector_type)

    if descriptor_type is "HOG":
        winSize = (16,16)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        winStride = (8,8)
        padding = (8,8)
        descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    elif descriptor_type is "SURF":
        thresh = 500
        highQuality = 1
        descriptor = cv2.SURF(thresh, 4, 2, highQuality, 1)
    else:
        descriptor = cv2.DescriptorExtractor_create(descriptor_type)

    K=[]
    D=[]
    des = list()
    print 'Extracting Local Descriptors'
    print "length filenames: "+str(len(filenames))

    init=time.time()

    for filename in filenames:
        ima=cv2.imread(filename)
        if color_constancy:
            ima = YCRCBhistogramEqualization(ima)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpts = detector.detect(gray)
        # kpts = sorted(kpts, key=getScoreKeyPoint,reverse=True)
        # kpointperimage = int(num_samples/len(filenames))
        # kpointperimage = min(kpointperimage, len(kpts)-1)
        # kpts = kpts[0:kpointperimage]
        if descriptor_type is "HOG":
            padded_im = cv2.copyMakeBorder(gray,32,32,32,32,cv2.BORDER_REFLECT)

            for i in range(0,len(kpts)):
                #get coordinates of SIFT feature from keypoint attribute "pt", which is a tuple with coordinates (x,y)
                a = np.round(kpts[i].pt[0]+32)
                b = np.round(kpts[i].pt[1]+32)

                loc = ((a,b),)
                temp = descriptor.compute(padded_im,winSize,padding,loc)
                des.append(temp)
                des = np.asarray(des)
                des = des.reshape((des.shape[0],des.shape[1]))

        elif descriptor_type is 'LBP':
            image = gray
            R = 1
            P = R * 8
            des = feat.local_binary_pattern(image, P, R, 'uniform')
        else:
            kpts, des = descriptor.compute(gray, kpts)

        K.append(kpts)
        D.append(des)
        des = list()

    ind2remove = []  #indices to remove

    for kpts_index in xrange(len(K)):
        if (len(K[kpts_index]) == 0):
            ind2remove.append(kpts_index)

    count = 0
    for i in ind2remove:
        K.remove(K[i+count])
        D.remove(D[i+count])
        filenames.remove(filenames[i])
        GT_ids.remove(GT_ids[i])
        GT_labels.remove(GT_labels[i])
        count = count - 1


    end=time.time()
    print 'Done in '+str(end-init)+' secs.'

    # Covariance of the descriptors
    # img = D[0]
    # img_w = D_w[0]
    # cov_img = np.cov(img)
    # cov_img_w = np.cov(img_w)
    # plt.imshow(cov_img)
    # plt.imshow(cov_img_w)

    return(K,D,filenames, GT_ids, GT_labels)


def saveDSCRepresentation(DSC_train): # save plots of first 30 elements of D (#blobs x #feats)
    i = 0
    while (i < 30):
        img = DSC_train[i]
        plt.imshow(img, cmap='Paired', aspect='auto')
        plt.savefig('DSC' + str(i) + 'whiten.png')
        i = i + 1

def getLocalColorDescriptors(filenames, keypoints, color_constancy):
    CD = []
    area = 4
    n_bins = 16
    print 'Extracting Local Color Descriptors'
    init = time.time()
    cont = 0
    for filename in filenames:
        kpts = keypoints[cont]
        cdesc = np.zeros((len(kpts), n_bins), dtype=np.float32)
        ima = cv2.imread(filename)
        if color_constancy:
            ima = YCRCBhistogramEqualization(ima)
        hls = cv2.cvtColor(ima, cv2.COLOR_BGR2HLS)
        hue = hls[:, :, 0]
        w, h = hue.shape
        cont2 = 0
        for k in kpts:
            patch = hue[max(0, k.pt[0] - area * k.size):min(w, k.pt[0] + area * k.size),
                    max(0, k.pt[1] - area * k.size):min(h, k.pt[1] + area * k.size)]
            hist, bin_edges = np.histogram(patch, bins=n_bins, range=(0, 180))
            cdesc[cont2, :] = hist
            cont2 += 1
        cont += 1
        CD.append(cdesc)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return (CD)

def getLocalColorProbabilities(filenames, keypoints, color_constancy):
    CD = []
    cont = 0
    for filename in filenames:
        kpts = keypoints[cont]
        ima = cv2.imread(filename)
        if color_constancy:
            ima = YCRCBhistogramEqualization(ima)
        y_vector = []
        x_vector = []
        for k in range(len(kpts)):
            x_vector.append(int(kpts[k].pt[0]))
            y_vector.append(int(kpts[k].pt[1]))
        kptsCoordinates = [y_vector, x_vector]
        cdesc = ImColorNamingTSELabDescriptor(ima, np.column_stack((y_vector, x_vector)))
        cont += 1
        CD.append(cdesc)
    return (CD)

def ImColorNamingTSELabDescriptor(ima, positions=None, patchSize=1):

    # Constants
    numColors=11                               # Number of colors
    numAchromatics=3                           # Number of achromatic colors
    numChromatics=numColors-numAchromatics     # Number of chromatic colors

    # Initializations
    numLevels = np.size(thrL)-1                   # Number of Lightness levels in the model

    # Image conversion: sRGB to CIELab
    Lab = ImsRGB2Lab(ima)

    if positions!=None:
        if patchSize==1:
            Lab = Lab[positions[:,0],positions[:,1],:].reshape((1,-1,3))
        else:
            LabPatch = np.zeros((positions.shape[0],(2*np.trunc(patchSize/2)+1)**2,3))
            padSz = (int(np.trunc(patchSize/2)),int(np.trunc(patchSize/2)))
            Lab = pad(Lab,(padSz,padSz,(0,0)), 'symmetric')
            positions += padSz[0]
            c=0
            for x in range(-padSz[0],padSz[0]+1):
                for y in range(-padSz[0],padSz[0]+1):
                    LabPatch[:,c,:]=Lab[positions[:,0]+y,positions[:,1]+x,:]
                    c += 1
            Lab=LabPatch


    L=Lab[:,:,0].flatten()
    a=Lab[:,:,1].flatten()
    b=Lab[:,:,2].flatten()

    S = np.shape(Lab)
    nr = S[0]; nc = S[1];                       # Image dimensions: rows, columns, and channels
    npx = nr*nc                                 # Number of pixels
    CD = np.zeros((npx,numColors))              # Color descriptor to store results

    # Assignment of the sample to its corresponding level
    m = np.zeros(np.shape(L))
    m[np.where(L==0)[0]] = 1    # Pixels with L=0 assigned to level 1
    for k in range(1,numLevels+1):
        m = m + np.double(thrL[k-1]<L) * np.double(L<=thrL[k]) * np.double(k)

    m = m.astype(int) - 1

    # Computing membership values to chromatic categories
    for k in range(numChromatics):
        tx=np.reshape(parameters[k,0,m],(npx,1))
        ty=np.reshape(parameters[k,1,m],(npx,1))
        alfa_x=np.reshape(parameters[k,2,m],(npx,1))
        alfa_y=np.reshape(parameters[k,3,m],(npx,1))
        beta_x=np.reshape(parameters[k,4,m],(npx,1))
        beta_y=np.reshape(parameters[k,5,m],(npx,1))
        beta_e=np.reshape(parameters[k,6,m],(npx,1))
        ex=np.reshape(parameters[k,7,m],(npx,1))
        ey=np.reshape(parameters[k,8,m],(npx,1))
        angle_e=np.reshape(parameters[k,9,m],(npx,1)); #figure;plot(angle_e); show()
        CD[:,k] = (np.double(beta_e!=0.0) * TripleSigmoid_E(np.vstack((a,b)),tx,ty,alfa_x,alfa_y,beta_x,beta_y,beta_e,ex,ey,angle_e)).T

    # Computing membership values to achromatic categories
    valueAchro = np.squeeze(np.maximum(1.0-np.reshape(np.sum(CD,axis=1),(npx,1)),np.zeros((npx,1))))
    CD[:,numChromatics+0] = valueAchro * Sigmoid(L,paramsAchro[0,0],paramsAchro[0,1])
    CD[:,numChromatics+1] = valueAchro * Sigmoid(L,paramsAchro[1,0],paramsAchro[1,1])*Sigmoid(L,paramsAchro[2,0],paramsAchro[2,1])
    CD[:,numChromatics+2] = valueAchro * Sigmoid(L,paramsAchro[3,0],paramsAchro[3,1])


    # Color descriptor with color memberships to all the categories (one color in each plane)
    if positions == None or patchSize>1:
        CD = np.reshape(CD,(nr,nc,numColors))
    if patchSize>1:
        CD=np.sum(CD,axis=1)
        CD=CD/np.tile(np.sum(CD,axis=1).reshape(-1,1),(1,numColors))


    return CD

# ***********************
#    Sigmoid
# ***********************
def Sigmoid(s,t,b):

    y = 1.0/(1.0+np.exp(-np.double(b)*(np.double(s)-np.double(t))))

    return y



# ***********************
#    TripleSigmoid_E
# ***********************
def TripleSigmoid_E(s,tx,ty,alfa_x,alfa_y,bx,by,be,ex,ey,angle_e):

    sT = np.double(s.T) - np.hstack([tx, ty])
    sR = np.hstack([sT[:,0].reshape((-1,1))*np.cos(alfa_y)+sT[:,1].reshape((-1,1))*np.sin(alfa_y),\
                -sT[:,0].reshape((-1,1))*np.sin(alfa_x)+sT[:,1].reshape((-1,1))*np.cos(alfa_x)])
    sRE= np.hstack([sT[:,0].reshape((-1,1))*np.cos(angle_e)+sT[:,1].reshape((-1,1))*np.sin(angle_e),\
                -sT[:,0].reshape((-1,1))*np.sin(angle_e)+sT[:,1].reshape((-1,1))*np.cos(angle_e)])
    ex = (ex==0.0) + ex
    ey = (ey==0.0) + ey

    y = 1.0/np.hstack([1.0+np.exp(-sR*np.hstack([by, bx])),\
                    1.0+np.exp(-be*(np.sum((sRE/np.hstack([ex, ey]))**2.0,axis=1).reshape((-1,1))-1.0))])

    return np.prod(y,axis=1).reshape((-1,1))

def ImsRGB2Lab(Ima):

    # RGB > XYZ transformation matrix (sRGB with D65)
    M = np.vstack(([0.412424, 0.357579, 0.180464],[0.212656, 0.715158, 0.0721856],[0.0193324, 0.119193, 0.950444]))
    Xn = 0.9505; Yn = 1.0000; Zn = 1.0891;

    Ima = Ima/255.0
    S = np.shape(Ima)
    NF = S[0]; NC = S[1]; NCh = S[2]

    lRGB = np.zeros((3,1))
    XYZ = np.zeros((3,1))
    ImaLab = np.zeros((NF, NC, NCh))

    fRGB = np.vstack((np.reshape(Ima[:,:,0].T,(1,NF*NC)),np.reshape(Ima[:,:,1].T,(1,NF*NC)),np.reshape(Ima[:,:,2].T,(1,NF*NC))))
    lRGB = (fRGB<=0.04045)*(fRGB/12.92)+(fRGB>0.04045)*(((fRGB+0.055)/1.055)**2.4)
    XYZ = np.dot(M,lRGB)

    f_X2 = (XYZ[0]/Xn > 0.008856)*((XYZ[0]/Xn)**(1.0/3.0))+(XYZ[0]/Xn <= 0.008856)*(7.787*(XYZ[0]/Xn)+(16.0/116.0))
    f_Y2 = (XYZ[1]/Yn > 0.008856)*((XYZ[1]/Yn)**(1.0/3.0))+(XYZ[1]/Yn <= 0.008856)*(7.787*(XYZ[1]/Yn)+(16.0/116.0))
    f_Z2 = (XYZ[2]/Zn > 0.008856)*((XYZ[2]/Zn)**(1.0/3.0))+(XYZ[2]/Zn <= 0.008856)*(7.787*(XYZ[2]/Zn)+(16.0/116.0))

    L2  = (XYZ[1]/Yn > 0.008856)*((116.0*((XYZ[1]/Yn)**(1.0/3.0)))-16.0)+(XYZ[1]/Yn <= 0.008856)*(903.3*(XYZ[1]/Yn))
    a2 = 500.0*(f_X2-f_Y2)
    b2 = 200.0*(f_Y2-f_Z2)

    ImaLab[:,:,0] = np.reshape(L2,(NC,NF)).T
    ImaLab[:,:,1] = np.reshape(a2,(NC,NF)).T
    ImaLab[:,:,2] = np.reshape(b2,(NC,NF)).T

    return ImaLab


def getAndSaveCodebook(descriptors, num_samples, k, filename):
    size_descriptors = descriptors[0].shape[1]
    A = np.zeros((num_samples, size_descriptors), dtype=np.float32)
    # A = []
    ##Random choice of decriptors -- Dont do that!
    for i in range(num_samples):
    	A[i,:]= random.choice(random.choice(descriptors))

    ##Random choice of num_samples/num_images of descriptors from each image
    # descriptors_per_image = int(num_samples / descriptors.__len__())
    # descriptor_index = 0
    # for i in range(descriptors.__len__()):
    #     for j in range(descriptors_per_image):
    #         A[descriptor_index, :] = random.choice(descriptors[i])
    #         descriptor_index += 1

    ##Take all descriptors
    descriptor_index = 0
    descriptors_per_image = int(num_samples / descriptors.__len__())

    # for i in range(descriptors.__len__()):
    #  for j in range(descriptors[i].__len__()):
    #     A[descriptor_index,:]= descriptors[i][j]
    #     descriptor_index += 1
    # for i in range(descriptors.__len__()):
    #     for j in range(descriptors[i].__len__()):
    #         A.append(descriptors[i][j])


    print 'Computing kmeans on ' + str(num_samples) + ' samples with ' + str(k) + ' centroids'
    init = time.time()

    codebook, v = vq.kmeans(A, k, 3)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    cPickle.dump(codebook, open(filename, "wb"))
    return codebook


def getAndSaveCodebookSoftAsignement(descriptors, k):

    A = []
    for i in range(descriptors.__len__()):
        for j in range(descriptors[i].__len__()):
            A.append(descriptors[i][j])

    print 'Computing kmeans on samples with ' + str(k) + ' centroids'
    init = time.time()

    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(A)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'

    return kmeans


def getAndSaveBoVWRepresentation(descriptors, k, codebook, filename):
    print 'Extracting visual word representations'
    init = time.time()
    visual_words = np.zeros((len(descriptors), k), dtype=np.float32)
    for i in xrange(len(descriptors)):
        words, distance = vq.vq(descriptors[i], codebook)
        visual_words[i, :] = np.bincount(words, minlength=k)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    cPickle.dump(visual_words, open(filename, "wb"))
    return visual_words

def getAndSaveBoVWRepresentationSoftAssignement(descriptors, k, codebook, R, alpha):
    #R: Maximum number of centroids to consider
    #Alpha: Hight alpha --> Centroids with hight distance do not receibe a bin
    print 'Extracting visual word representations using soft margin'
    init = time.time()
    visual_words = np.zeros((len(descriptors), k), dtype=np.float32)
    for i in xrange(len(descriptors)):
        curImageDescriptors = descriptors[i]
        #For each keypoint descriptor of every image, select R nearest centroids and weight them in function of its distance
        for d in xrange(len(curImageDescriptors)):
            imageWords = np.empty(0,dtype=np.int)
            curDescriptor = curImageDescriptors[d:d+1]

            #Compute distances to centroids
            distances = codebook.transform(curDescriptor)
            #Sort centroids by distance
            sort_index = np.argsort(distances)
            #Compute weight of every word
            w = np.zeros(R)
            for r in range (0,R):
                w[r] = math.exp(-(distances[0][sort_index[0][r]]**2 / (2 * alpha **2)))

            #We assign every descriptor 5 word slots to spread in  R possible words (instead of only the closest one).
            wSum = sum(w)
            descriptorWords2Add = np.empty(0,dtype=np.int)
            for r in range (0,R):
                #Compute word corresponding slots repetitions
                w[r] = (w[r] * 5) / wSum
                #Create a vector with word repetitions and add it to descriptor's words
                curWord = np.empty(int(round(w[r])),dtype=np.int)
                curWord.fill(sort_index[0][r])
                descriptorWords2Add = np.concatenate([descriptorWords2Add, curWord])

            #Add 10 words of current descriptors to image words
            print descriptorWords2Add
            imageWords = np.concatenate([imageWords, descriptorWords2Add])

        print 'New IMAGEEEEEEEEEEEES'
        #L1 normalization of image words histogram
        hist = np.bincount(imageWords, minlength=k)
        histSum = sum(hist)
        for b in xrange(len(hist)):
         hist[b] = hist[b] / histSum

        #Add image words histogram to visual words
        visual_words[i, :] = hist

    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return visual_words


def getAndSaveBoVW_SPMRepresentation(descriptors, keypoints, k, codebook, filename, files, config):
    print 'Extracting visual word representations with SPM'
    init = time.time()

    if config is 1:
        visual_words = np.zeros((len(descriptors), k * 21), dtype=np.float32)

    elif config is 2:
        visual_words = np.zeros((len(descriptors), k * 4), dtype=np.float32)

    elif config is 3:
        visual_words = np.zeros((len(descriptors), k * 10), dtype=np.float32)

    elif config is 4:
        visual_words = np.zeros((len(descriptors), k * 5), dtype=np.float32)



    #For each image
    for i in xrange(len(descriptors)):

        #Read image and take its size
        ima = cv2.imread(files[i])
        w, h, _ = ima.shape

        #Assign the code of the closest word to the image descriptors
        words, distance = vq.vq(descriptors[i], codebook)

        #Now if we wanted to consider the whole image we could count word repetitions (np.bincount) to build the histogram
        #But we want to consider different regions.

        if config is 1: #I+2x2+4x4 grid

            #Upper-left
            idx_bin1 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 2)) & (x[0] < (1 * w / 2)) & (x[1] >= (0 * h / 2)) & (x[1] < (1 * h / 2)))]
            #Upper-right
            idx_bin2 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 2)) & (x[0] < (2 * w / 2)) & (x[1] >= (0 * h / 2)) & (x[1] < (1 * h / 2)))]
            #Bottom-left
            idx_bin3 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 2)) & (x[0] < (1 * w / 2)) & (x[1] >= (1 * h / 2)) & (x[1] < (2 * h / 2)))]
            #Bottom-right
            idx_bin4 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 2)) & (x[0] < (2 * w / 2)) & (x[1] >= (1 * h / 2)) & (x[1] < (2 * h / 2)))]

            #4x4 Grid: 16 Regions
            idx_bin5 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 4)) & (x[0] < (1 * w / 4)) & (x[1] >= (0 * h / 4)) & (x[1] < (1 * h / 4)))]
            idx_bin6 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 4)) & (x[0] < (2 * w / 4)) & (x[1] >= (0 * h / 4)) & (x[1] < (1 * h / 4)))]
            idx_bin7 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (2 * w / 4)) & (x[0] < (3 * w / 4)) & (x[1] >= (0 * h / 4)) & (x[1] < (1 * h / 4)))]
            idx_bin8 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (3 * w / 4)) & (x[0] < (4 * w / 4)) & (x[1] >= (0 * h / 4)) & (x[1] < (1 * h / 4)))]

            idx_bin9 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 4)) & (x[0] < (1 * w / 4)) & (x[1] >= (1 * h / 4)) & (x[1] < (2 * h / 4)))]
            idx_bin10 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (1 * w / 4)) & (x[0] < (2 * w / 4)) & (x[1] >= (1 * h / 4)) & (x[1] < (2 * h / 4)))]
            idx_bin11 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (2 * w / 4)) & (x[0] < (3 * w / 4)) & (x[1] >= (1 * h / 4)) & (x[1] < (2 * h / 4)))]
            idx_bin12 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (3 * w / 4)) & (x[0] < (4 * w / 4)) & (x[1] >= (1 * h / 4)) & (x[1] < (2 * h / 4)))]

            idx_bin13 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (0 * w / 4)) & (x[0] < (1 * w / 4)) & (x[1] >= (2 * h / 4)) & (x[1] < (3 * h / 4)))]
            idx_bin14 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (1 * w / 4)) & (x[0] < (2 * w / 4)) & (x[1] >= (2 * h / 4)) & (x[1] < (3 * h / 4)))]
            idx_bin15 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (2 * w / 4)) & (x[0] < (3 * w / 4)) & (x[1] >= (2 * h / 4)) & (x[1] < (3 * h / 4)))]
            idx_bin16 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (3 * w / 4)) & (x[0] < (4 * w / 4)) & (x[1] >= (2 * h / 4)) & (x[1] < (3 * h / 4)))]

            idx_bin17 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (0 * w / 4)) & (x[0] < (1 * w / 4)) & (x[1] >= (3 * h / 4)) & (x[1] < (4 * h / 4)))]
            idx_bin18 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (1 * w / 4)) & (x[0] < (2 * w / 4)) & (x[1] >= (3 * h / 4)) & (x[1] < (4 * h / 4)))]
            idx_bin19 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (2 * w / 4)) & (x[0] < (3 * w / 4)) & (x[1] >= (3 * h / 4)) & (x[1] < (4 * h / 4)))]
            idx_bin20 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                         ((x[0] >= (3 * w / 4)) & (x[0] < (4 * w / 4)) & (x[1] >= (3 * h / 4)) & (x[1] < (4 * h / 4)))]


            #Now we count word repetitions in each region to build 21 histograms (Whole image + 20 regions), and we concatenate them
            visual_words[i, :] = np.hstack((np.bincount(words, minlength=k), np.bincount(words[idx_bin1], minlength=k),
                                            np.bincount(words[idx_bin2], minlength=k),
                                            np.bincount(words[idx_bin3], minlength=k),
                                            np.bincount(words[idx_bin4], minlength=k),
                                            np.bincount(words[idx_bin5], minlength=k),
                                            np.bincount(words[idx_bin6], minlength=k),
                                            np.bincount(words[idx_bin7], minlength=k),
                                            np.bincount(words[idx_bin8], minlength=k),
                                            np.bincount(words[idx_bin9], minlength=k),
                                            np.bincount(words[idx_bin10], minlength=k),
                                            np.bincount(words[idx_bin11], minlength=k),
                                            np.bincount(words[idx_bin12], minlength=k),
                                            np.bincount(words[idx_bin13], minlength=k),
                                            np.bincount(words[idx_bin14], minlength=k),
                                            np.bincount(words[idx_bin15], minlength=k),
                                            np.bincount(words[idx_bin16], minlength=k),
                                            np.bincount(words[idx_bin17], minlength=k),
                                            np.bincount(words[idx_bin18], minlength=k),
                                            np.bincount(words[idx_bin19], minlength=k),
                                            np.bincount(words[idx_bin20], minlength=k)))

        elif config is 2: #I + 3x1 grid

            #Upper
            idx_bin1 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[1] >= (0 * h / 3)) & (x[1] < (1 * h / 3)))]
            #Center
            idx_bin2 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[1] >= (1 * h / 3)) & (x[1] < (2 * h / 3)))]
            #Bottom
            idx_bin3 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[1] >= (2 * h / 3)) & (x[1] < (3 * h / 3)))]

            visual_words[i, :] = np.hstack((np.bincount(words, minlength=k), np.bincount(words[idx_bin1], minlength=k),
                                            np.bincount(words[idx_bin2], minlength=k),
                                            np.bincount(words[idx_bin3], minlength=k)))


        elif config is 3: #I+3x3 grid

            idx_bin1 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 3)) & (x[0] < (1 * w / 3)) & (x[1] >= (0 * h / 3)) & (x[1] < (1 * h / 3)))]
            idx_bin2 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 3)) & (x[0] < (2 * w / 3)) & (x[1] >= (0 * h / 3)) & (x[1] < (1 * h / 3)))]
            idx_bin3 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 3)) & (x[0] < (1 * w / 3)) & (x[1] >= (1 * h / 3)) & (x[1] < (2 * h / 3)))]
            idx_bin4 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 3)) & (x[0] < (2 * w / 3)) & (x[1] >= (1 * h / 3)) & (x[1] < (2 * h / 3)))]
            idx_bin5 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (2 * w / 3)) & (x[0] < (3 * w / 3)) & (x[1] >= (0 * h / 3)) & (x[1] < (1 * h / 3)))]
            idx_bin6 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (2 * w / 3)) & (x[0] < (3 * w / 3)) & (x[1] >= (1 * h / 3)) & (x[1] < (2 * h / 3)))]
            idx_bin7 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 3)) & (x[0] < (1 * w / 3)) & (x[1] >= (2 * h / 3)) & (x[1] < (3 * h / 3)))]
            idx_bin8 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 3)) & (x[0] < (2 * w / 3)) & (x[1] >= (2 * h / 3)) & (x[1] < (3 * h / 3)))]
            idx_bin9 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (2 * w / 3)) & (x[0] < (3 * w / 3)) & (x[1] >= (2 * h / 3)) & (x[1] < (3 * h / 3)))]


            visual_words[i, :] = np.hstack((np.bincount(words, minlength=k), np.bincount(words[idx_bin1], minlength=k),
                                            np.bincount(words[idx_bin2], minlength=k),
                                            np.bincount(words[idx_bin3], minlength=k),
                                            np.bincount(words[idx_bin4], minlength=k),
                                            np.bincount(words[idx_bin5], minlength=k),
                                            np.bincount(words[idx_bin6], minlength=k),
                                            np.bincount(words[idx_bin7], minlength=k),
                                            np.bincount(words[idx_bin8], minlength=k),
                                            np.bincount(words[idx_bin9], minlength=k)))


        elif config is 4: #I+2x2 grid

            #Upper-left
            idx_bin1 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 2)) & (x[0] < (1 * w / 2)) & (x[1] >= (0 * h / 2)) & (x[1] < (1 * h / 2)))]
            #Upper-right
            idx_bin2 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 2)) & (x[0] < (2 * w / 2)) & (x[1] >= (0 * h / 2)) & (x[1] < (1 * h / 2)))]
            #Bottom-left
            idx_bin3 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (0 * w / 2)) & (x[0] < (1 * w / 2)) & (x[1] >= (1 * h / 2)) & (x[1] < (2 * h / 2)))]
            #Bottom-right
            idx_bin4 = [j for j, x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if
                        ((x[0] >= (1 * w / 2)) & (x[0] < (2 * w / 2)) & (x[1] >= (1 * h / 2)) & (x[1] < (2 * h / 2)))]

            visual_words[i, :] = np.hstack((np.bincount(words, minlength=k), np.bincount(words[idx_bin1], minlength=k),
                                            np.bincount(words[idx_bin2], minlength=k),
                                            np.bincount(words[idx_bin3], minlength=k),
                                            np.bincount(words[idx_bin4], minlength=k)))

    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    cPickle.dump(visual_words, open(filename, "wb"))
    return visual_words


def trainAndTestLinearSVM(train, test, GT_train, GT_test, c):  # TODO: plot data in 2D feature space for visualization
    print 'Training and Testing a linear SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    test = stdSlr.transform(test)
    clf = svm.SVC(kernel='linear', C=c,  probability=True).fit(train, GT_train)

    # Compute mean accuracy
    accuracy = 100 * clf.score(test, GT_test)
    predictions = clf.predict(test)
    predict_prob_train = clf.predict_proba(train)
    predict_prob_test = clf.predict_proba(test)

    # Classify samples -Added (R)
    decision_function = clf.decision_function(test)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, predictions, decision_function, predict_prob_train, predict_prob_test


def trainAndTestRBFSVM(train, test, GT_train, GT_test, c):  # TODO: optimize gamma and c
    print 'Training and Testing SVM with RBF Kernel'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)

    clf = svm.SVC(kernel='rbf', C=c).fit(train, GT_train) # gamma=0.01 ???
    accuracy = 100 * clf.score(stdSlr.transform(test), GT_test)
    predictions = clf.predict(stdSlr.transform(test)) #proba check it out!!!!! To compute the ROC
    decision_function = clf.decision_function(test)

    # predict_prob_train = clf.predict_proba(train)
    # predict_prob_test = clf.predict_proba(test)

    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, predictions, decision_function, 0, 0


def trainAndTestPolySVM(train, test, GT_train, GT_test, c):  # TODO: optimize degree and coef0
    print 'Training and Testing a Polynomial SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    clf = svm.SVC(kernel='poly', C=c, degree=3, coef0=0.0).fit(train, GT_train)
    accuracy = 100 * clf.score(stdSlr.transform(test), GT_test)

    decision_function = clf.decision_function(stdSlr.transform(test))
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, decision_function


def trainAndTestSigSVM(train, test, GT_train, GT_test, c):  # TODO: optimize coef0
    print 'Training and Testing a Sigmoid SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    clf = svm.SVC(kernel='sigmoid', C=c, degree=3, coef0=0.0).fit(train, GT_train)
    accuracy = 100 * clf.score(stdSlr.transform(test), GT_test)

    decision_function = clf.decision_function(stdSlr.transform(test))
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, decision_function


def trainAndTestLinearSVM_withfolds(train, test, GT_train, GT_test, folds):
    print 'Training and Testing a Linear SVM with folds'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    test = stdSlr.transform(test)
    kernelMatrix = histogramIntersection(train, train)

    tuned_parameters = [{'kernel': ['linear'], 'C': np.linspace(0.01, 3, num=15)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds, scoring='accuracy')
    clf.fit(kernelMatrix, GT_train)

    print clf.best_params_
    predictMatrix = histogramIntersection(test, train)
    SVMpredictions = clf.predict(predictMatrix)
    predict_prob_train = clf.predict_proba(kernelMatrix)
    predict_prob_test = clf.predict_proba(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = 100 * (correct / len(GT_test))

    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, SVMpredictions, predict_prob_train, predict_prob_test

def trainAndTestRBFSVM_withfolds(train, test, GT_train, GT_test, folds):  # TODO: optimize gamma and c
    print 'Training and Testing SVM with RBF Kernel with folds'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    test = stdSlr.transform(test)
    kernelMatrix = histogramIntersection(train, train)

    tuned_parameters = [{'kernel': ['rbf'], 'C': np.linspace(0.01, 3, num=15)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds, scoring='accuracy')
    clf.fit(kernelMatrix, GT_train)
    print clf.best_params_

    predictMatrix = histogramIntersection(test, train)
    predictions = clf.predict(predictMatrix)

    # predict_prob_train = clf.predict_proba(kernelMatrix)
    # predict_prob_test = clf.predict_proba(predictMatrix)

    correct = sum(1.0 * (predictions == GT_test))
    accuracy = 100 * (correct / len(GT_test))

    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, predictions, 0, 0

def trainAndTestChi2SVM(train, test, GT_train, GT_test, c):
    print 'Training and Testing a Chi2Kernel SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    test = stdSlr.transform(test)
    train = train + train.min().__abs__()
    test = test + test.min().__abs__()

    kernelMatrix = additive_chi2_kernel(train, train)
    clf = svm.SVC(kernel='precomputed', C=c)
    clf.fit(kernelMatrix, GT_train)
    predictMatrix = additive_chi2_kernel(test, train)

    decision_function = clf.decision_function(predictMatrix)
    SVMpredictions = clf.predict(predictMatrix)

    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = 100 * (correct / len(GT_test))

    # predict_prob_train = clf.predict_proba(kernelMatrix)
    # predict_prob_test = clf.predict_proba(predictMatrix)

    end = time.time()
    print 'Accuracy ' + str(accuracy)
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, SVMpredictions, decision_function, 0, 0


def trainAndTestChi2SVM_withfolds(train, test, GT_train, GT_test, folds):
    print 'Training and Testing a Chi2Kernel SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    test = stdSlr.transform(test)
    train = train + train.min().__abs__()
    test = test + test.min().__abs__()

    kernelMatrix = additive_chi2_kernel(train, train)

    tuned_parameters = [{'kernel': ['precomputed'], 'C': np.linspace(0.01, 3, num=15)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds, scoring='accuracy')
    clf.fit(kernelMatrix, GT_train)

    print(clf.best_params_)
    predictMatrix = additive_chi2_kernel(test, train)
    SVMpredictions = clf.predict(predictMatrix)

    # predict_prob_train = clf.predict_proba(kernelMatrix)
    # predict_prob_test = clf.predict_proba(predictMatrix)

    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = 100 * (correct / len(GT_test))
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy, SVMpredictions, 0, 0


def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]
    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp
    return result


def SPMKernel(M, N, k):
    m = M.shape[0]
    n = N.shape[0]

    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            temp = ((.25 * np.sum(np.minimum(M[i, 0:k], N[j, 0:k]))) + (
            .25 * np.sum(np.minimum(M[i, k:k * 5], N[j, k:k * 5]))) + (
                    .5 * np.sum(np.minimum(M[i, k * 5:k * 21], N[j, k * 5:k * 21]))))
            result[i][j] = temp
    return result


def trainAndTestHISVM(train, test, GT_train, GT_test, c):
    print 'Training and Testing a HI SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = histogramIntersection(train, train)
    clf = svm.SVC(kernel='precomputed', C=c)
    clf.fit(kernelMatrix, GT_train)
    predictMatrix = histogramIntersection(stdSlr.transform(test), train)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy


def trainAndTestHISVM_withfolds(train, test, GT_train, GT_test, folds):
    print 'Training and Testing a HI SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = histogramIntersection(train, train)
    tuned_parameters = [{'kernel': ['precomputed'], 'C': np.linspace(0.0001, 0.2, num=10)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds, scoring='accuracy')
    clf.fit(kernelMatrix, GT_train)
    print(clf.best_params_)
    predictMatrix = histogramIntersection(stdSlr.transform(test), train)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy


def trainAndTestSPMSVM(train, test, GT_train, GT_test, c, k):
    print 'Training and Testing a SPMKernel SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = SPMKernel(train, train, k)
    clf = svm.SVC(kernel='precomputed', C=c)
    clf.fit(kernelMatrix, GT_train)
    predictMatrix = SPMKernel(stdSlr.transform(test), train, k)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy


def trainAndTestSPMSVM_withfolds(train, test, GT_train, GT_test, k, folds):
    print 'Training and Testing a SPMKernel SVM'
    init = time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = SPMKernel(train, train, k)
    tuned_parameters = [{'kernel': ['precomputed'], 'C': np.linspace(0.0001, 0.2, num=10)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds, scoring='accuracy')
    clf.fit(kernelMatrix, GT_train)
    print(clf.best_params_)
    predictMatrix = SPMKernel(stdSlr.transform(test), train, k)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return accuracy

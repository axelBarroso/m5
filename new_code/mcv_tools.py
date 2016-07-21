""" 
    
    Library of tools for machine learning and data processing
    
     Master in Computer Vision - Barcelona
      
     Author: Francesco Ciompi, Ramon Baldrich, Jordi Gonzalez
         
"""

# Master in Computer Vision Libraries
import mcv_descriptors as dr

# External Libraries
import os                            # operating system
from numpy import *                    # numpy, for maths computations
import numpy as np
import random as rnd                # random value generator
import cv2
from skimage.feature import local_binary_pattern
import multiprocessing as mp
from sklearn.decomposition import PCA
#import platform
#if not platform.node().lower().startswith('compute'):
from pylab import *       # matplotlib, for graphical plots
from PIL import Image


#
#     get list of files in 'path' with extension 'ext'
#
def getFilelist(path,ext='',init=''):
    # returns a list of filenames (absolute, relative) for all 'ext' images in a directory
    if ext != '':
        file_list = [f for f in os.listdir(path) if f.startswith(init) and f.endswith('.'+ext)]
        return [os.path.join(path,f) for f in file_list], file_list
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]    


#
#     subsample over the rows of data matrix 'data'
#
def Subsample(data, nPt):
        
    if shape(data)[0] > nPt:
        #any better option?
        idx = array(rnd.sample(range(0,shape(data)[0]), nPt));
        return data[idx]
    else:
        return data


#==============================================================================
# #
# #     Compute feature matrix per list of images
# #
def getFeatList(FileList, params, runmultiprocess, showFig):
    img = cv2.imread(FileList[0])
    feats, im_idx = getFeatImgSample(img,params,0,True,False)
    D = feats.shape[1]
    NR = len(FileList)*params['nPt']

    # initialize data matrix
    X = zeros((NR,D))
    idxs = zeros((len(FileList),2))


    if runmultiprocess: 
        pool = mp.Pool(mp.cpu_count()- runmultiprocess if mp.cpu_count()>1 else 0)
        PoolResults = []

    idx = 0
    n_im = -1
    for imfile in FileList:
        n_im += 1
        print imfile
        img = cv2.imread(imfile)

        if runmultiprocess:         ########### start change
            PoolResults.append(pool.apply_async(getFeatImgSample, args=(img,params,n_im,True,False)))    
        else:
            feats, im_idx = getFeatImgSample(img,params,n_im,True,showFig)
            if shape(feats)[0]>0:
                idxs[im_idx,0]=idx
                idxs[im_idx,1]=shape(feats)[0]
                X[idx:idx+shape(feats)[0]] = feats
                idx+=shape(feats)[0]

    if runmultiprocess:
        pool.close()
        while (len(PoolResults)>0):
            try:
                feats, im_idx = PoolResults[0].get(timeout=0.001)
                PoolResults.pop(0)
                
                if shape(feats)[0]>0:
                    idxs[im_idx,0]=idx
                    idxs[im_idx,1]=shape(feats)[0]
                    X[idx:idx+shape(feats)[0]] = feats
                    idx+=shape(feats)[0]
            except:
                pass
                
        pool.join()
        
    return X,idxs
#==============================================================================

def getFeatImgSample(img,params,n_im,subsample,showFig):
    feats,pos = getFeatImg(img,params,showFig)
    print n_im, feats.shape
    if subsample and feats is not None:
        feats = Subsample(feats, params['nPt'])
    return (feats,  n_im)
    
#
#     compute feature matrix per image
#
def getFeatImg(img, params, showFig=False, filename=''):
    """     Compute features corresponding to the selcted 'descriptor' for image 'img'
    The descriptor is applied to patches (patchSize) over a dense matrix (gridSpacing)
    in case of SIFT and HOG, and to keypoints in all the others. It also implements the
    early fusion as a combination of descriptors """
    #
    #    no keypoints: used in image retrieval and global descriptors
    #
    kp=[]
    #img = cv2.imread(filename)
    # Create detector
    # ["FAST","STAR","SIFT","SURF","ORB","MSER","GFTT","HARRIS","Dense"]
    opencvdic = {'fast': 'FAST', 'star': 'STAR', 'sift': 'SIFT', 'mser': 'MSER', 'gftt': 'GFTT',
                 'harris': 'HARRIS', 'dense': 'Dense', 'orb': 'ORB','surf':'SURF'}
    detector = cv2.FeatureDetector_create(opencvdic[params['keypointDetector']])
    # find the keypoints with STAR
    kp = detector.detect(img, None)

    if showFig:
        plt.figure(params['keypointDetector'])
        plt.imshow(img)
        plt.hold(True)
        for keyp in kp:
            x, y = keyp.pt
            plt.plot(x, y, 'ro')
            plt.axis('equal')
        plt.show()

    # LOCAL DESCRIPTORS

    if len(kp) < 1:
        print 'Image skipped for lack of key-points'
    else:
        DESCRIPTORS = []
        POSITIONS = []
        # ####################################
        #     Compute image descriptor(s)
        # ####################################


        opencvdic = {'brief': 'BRIEF', 'surf': 'SURF', 'sift': 'SIFT', 'hog': 'HOG', 'orb': 'ORB', 'freak': 'FREAK'}

        desnumber = 1
        for descriptor in params['descriptor']:
            # ##############################################
            #     SIFT, SURF, SIFT, BRIEF, ORB, FREAK
            # ##############################################

            desName= descriptor.replace("Opponent","")

            if desName in ['sift', 'brief', 'surf', 'orb', 'freak']:


                if "Opponent" in descriptor:
                    opencvDescriptorName= "Opponent"+opencvdic[desName]
                else:
                    opencvDescriptorName= opencvdic[desName]

                descriptor_extractor = cv2.DescriptorExtractor_create(opencvDescriptorName)

                if desnumber == 1:
                    kp, des = descriptor_extractor.compute(img, kp)
                    DESCRIPTORS = des
                else:
                    new_keypoints = []
                    new_des = []

                    #For each descriptor
                    for i in range(0, len(kp)):
                        kpp, des = descriptor_extractor.compute(img, [kp[i]])
                        if len(kpp) > 0:
                            new_keypoints.append(kp[i])
                            tmp = (DESCRIPTORS[i].tolist() + des[0].tolist())
                            new_des.append(np.array(tmp))
                    DESCRIPTORS=new_des
                    kp=new_keypoints
                desnumber += 1
            # ########################################
            #     Histogram of Oriented Gradient (HOG)
            # ########################################
            elif descriptor == 'hog':
                new_keypoints = []
                new_des = []
                if img.ndim==3:
                    grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    grayimage = img
                if len(kp) > 0:
                    for i in range(0, len(kp)):
                        r = kp[i].pt[0]
                        c = kp[i].pt[1]
                        r = int(r)
                        c = int(c)

                        patch = grayimage[r-int(50/2):r+int(50/2), c-int(50/2):c+int(50/2)]
                        if size(patch) == 50*50:  # discard incomplete patches
                            # hog is found in mcv_descriptors.py
                            #des = hog(patch).transpose()
                            des = dr.hog(patch).transpose()
                            new_keypoints.append(kp[i])
                            if desnumber == 1:
                                new_des.append(des)
                            else:
                                tmp=(DESCRIPTORS[i].tolist() + des.tolist())
                                new_des.append(np.array(tmp))

                DESCRIPTORS = np.array(new_des)
                kp = new_keypoints
                desnumber += 1

            # ########################################
            #     LBP
            # ########################################
            elif descriptor == 'lbp':
                print 'LBP ('+str(params['nf_lbp'])+' elems.)'
                new_keypoints = []
                new_des = []
                if img.ndim==3:
                    grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    grayimage = img
                if len(kp) > 0:
                    for i in range(0, len(kp)):
                        r = kp[i].pt[0]
                        c = kp[i].pt[1]
                        r=int(r)
                        c=int(c)

                        patch = grayimage[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                        if size(patch) == params['lbp_patch_size']*params['lbp_patch_size']: # discard incomplete patches

                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')

                            des,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            new_keypoints.append(kp[i])
                            if desnumber == 1:
                                new_des.append(des)
                            else:
                                tmp=(DESCRIPTORS[i].tolist() + des.tolist())
                                new_des.append(np.array(tmp))

                DESCRIPTORS = np.array(new_des)
                kp = new_keypoints
                desnumber += 1
            # ########################################
            #     LBP COLOR
            # ########################################
            elif descriptor == 'lbp_color':
                new_keypoints = []
                new_des = []
                if img.ndim==3:
                    grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    grayimage = img
                if len(kp) > 0:
                    for i in range(0, len(kp)):
                        r = kp[i].pt[0]
                        c = kp[i].pt[1]
                        r=int(r)
                        c=int(c)

                        patch = grayimage[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                        if size(patch) == params['lbp_patch_size']*params['lbp_patch_size']: # discard incomplete patches
                            blue, green, red = cv2.split(img)

                            patch = blue[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')
                            desb,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            patch = green[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')
                            desg,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            patch = red[r-int(params['lbp_patch_size']/2):r+int(params['lbp_patch_size']/2), c-int(params['lbp_patch_size']/2):c+int(params['lbp_patch_size']/2)]
                            lbp = local_binary_pattern(patch, 8, 1, 'nri_uniform')
                            desr,bins = np.histogram( lbp.ravel(),bins=59, range=(0, 59))

                            des = np.hstack((desb,desg,desr))
                            new_keypoints.append(kp[i])
                            if desnumber == 1:
                                new_des.append(des)
                            else:
                                tmp=(DESCRIPTORS[i].tolist() + des.tolist())
                                new_des.append(np.array(tmp))

                DESCRIPTORS = np.array(new_des)
                kp = new_keypoints
                desnumber += 1

    POSITIONS = getPositions(kp)
    if len(DESCRIPTORS) < 1:
        DESCRIPTORS=None
    return DESCRIPTORS, POSITIONS

def getPositions(keypoints):
    pos=[]
    for k in keypoints:
        pos.append(k.pt)
    return pos
#
#     normalize feature matrix
#
def normalizeFeat(x,mean_x=None,std_x=None):
    
    if mean_x == None and std_x is None:
        mean_x = x.mean(axis=0)
        std_x = x.std(axis=0)
        std_x[std_x==0] = 1
                
    return (x-tile(mean_x,(shape(x)[0],1)))/tile(std_x,(shape(x)[0],1)),mean_x,std_x
    
def getPCA(x,params):
    pca = None
    if params['pca']>0:
        pca = PCA(n_components=params['pca'], whiten=False)
        pca.fit(x)
    return pca
    #x = pca.fit_transform(x)
    
    
#    
#     accuracy of a confusion matrix    
#
def accuracy(confMat):
    
    return sum(diag(confMat))/sum(confMat.flatten())
    
    
#    
#     compute the row with minimum distance between vector 'v' and matrix 'm'
#
def yMinDist(v,m,metric):

    Dist = zeros((1,shape(m)[0]))
    
    for row in range(0,shape(m)[0]):
        
        if metric == 'euclidean':

            Dist[0,row] = sqrt(sum((m[row,:]-v)**2))
            
    return argmin(Dist[0,:])
    
        
#
#    Hamming distance
#
""" Calculate the Hamming distance between two given strings """
def hamming(a, b):
    
    return sum(logical_xor(a,b).astype(int))
    
    
from sklearn.neighbors import NearestNeighbors

def fastnn(queryDescriptors, trainDescriptors, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(trainDescriptors)
    distances, indices = nbrs.kneighbors(queryDescriptors)
    return indices[0]
    
def MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, k, args):
    hitmiss = checkHitsMiss(filename, trainDir, LIBRARY, idxs, k, args)
    map = 0
    accu = 0
    for k in range(0,len(hitmiss)):
        if hitmiss[k]==1:
            accu += 1
            map += double(accu)/double(k+1)
    return (map/accu) if accu >0 else 0
        
    
def checkHitsMiss(filename, trainDir, LIBRARY, idxs, k, args):
    hit_miss = zeros((k,1))
    isreallycar = (len(filename.split('car'))>1)
    for i in range(0,k):
        # iscar = os.path.isfile(trainDir+'car/'+LIBRARY[idxs[i]].split('/')[-1][3:-4]+'.JPEG')
        # hit_miss[i] = (isreallycar == iscar)

        if (args.qtechnique == 'fastnn'):
            iscar = os.path.isfile(trainDir+'car/'+LIBRARY[idxs[i]].split('/')[-1][3:-4]+'.JPEG')
            hit_miss[i] = (isreallycar == iscar)
        elif (args.qtechnique == 'pquant' and not args.lsh):
            iscar = os.path.isfile(trainDir+'car/'+LIBRARY[idxs[i]].split('/')[-1][6:-4]+'.JPEG')
            hit_miss[i] = (isreallycar == iscar)
        elif (args.lsh):
            iscar = os.path.isfile(trainDir+'car/'+LIBRARY[idxs[i]].split('/')[-1][9:-4]+'.JPEG')
            hit_miss[i] = (isreallycar == iscar)

    return hit_miss
   
#
#    show first 10 results of image-based query
#    
def showQueryResult(filename,trainDir,LIBRARY,idxs,method):
            
    #
    #    QUERY RESULTS VISUALIZATION (first 10 results)
    #                    
    ion()
    figure('IMAGE RETRIEVAL'); subplot(353); title('QUERY')
    img = array(Image.open(filename))
    imshow(img)
    axis('off')

    for i in range(0,10):
        subplot(3,5,6+i)
        try:
            img = array(Image.open(trainDir+'car/'+LIBRARY[idxs[i]].split('/')[-1][3:-4]+'.JPEG'))
        except:
            img = array(Image.open(trainDir+'dog/'+LIBRARY[idxs[i]].split('/')[-1][3:-4]+'.JPEG'))
        imshow(img)
        axis('off')
    
    draw()
    show(block=True)


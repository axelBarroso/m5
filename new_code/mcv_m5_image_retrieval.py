#!/bin/env python
'''
***************************************************************
CODE M5: Image rerieval

author: Francesco Ciompi
modified: Ramon Baldrich
***************************************************************
TASK
> retrieve images containing similar objects, based on a query

DATA
> ImageNet dataset
> 2 classes: dog, car
> ~ 5000 images per class
> query based on PASCAL images, similar images retrieved from ImageNet (large scale) library

METHODS
> Fisher vector 
> Product quantization

OPTIONS (implemented in product quantization)
> Locality Sensitive Hashing
***************************************************************
'''
import sys #omport from sys import exit ,path            # system functions for escape
import time
import os
sys.path.append('.')
import ArgParse
import pickle


#from PIL import Image     # python image library
#from sklearn.lda import *
#from numpy import *    # numpy, for maths computations
#if not platform.node().lower().startswith('compute'):
#    from pylab import *       # matplotlib, for graphical plots

import cv2         # opencv
import numpy as np    # numpy, for maths computations

import mcv_tools as tl
import mcv_bow as bw
import mcv_fisher as fv
import mcv_vlad as vl
#from mcv_pca import *    # principal component analysis    
import mcv_product_quantization as pq
import mcv_lsh as lsh
#from mcv_fastnn import *
#from scipy.sparse import *


# ******************  MAIN **************************
if __name__ == "__main__":
    #Parser-----------------------------------------------------------------------------------------------
    parser = ArgParse.ArgumentParser(description="Classification train", epilog="")
    parser.add_argument('-ds_folder',  dest='ds_folder',                    required=True,             help='Datasets folder')
    parser.add_argument('-out_folder', dest='out_folder',                   required=True,             help='Output folder were the results are stored')
    parser.add_argument('-detector',   dest='detector',   default='fast',   required=False,            help='Detector:{fast,star,sift,mser,gftt,harris,dense,orb,surf}')
    parser.add_argument('-descriptor', dest='descriptor', default=['sift'], required=False, nargs='+', help='Descriptor:{brief,surf,sift,hog,orb,freak}')
    parser.add_argument('-K',          dest='K',          default=200,      required=False, type=int,  help='Vocabulary size')
    parser.add_argument('-npt',        dest='npt',        default=50,       required=False, type=int,  help='Best n results')
    parser.add_argument('-pqK',        dest='pqK',        default=20,       required=False, type=int,  help='Best n results - product quantization (K kmeans)')
    parser.add_argument('-pqm',        dest='pqm',        default=8,        required=False, type=int,  help='pqm')
    parser.add_argument('-lsh',        dest='lsh',                          action='store_true',       help='LSH')
    parser.add_argument('-global',     dest='global_des', default='vlad',                              help='Global descriptor: bow|vlad|fisher vector')
    parser.add_argument('-getglobal',  dest='getglobal',                    action='store_true',       help='Compute global descriptor on trian images')
    parser.add_argument('-qtechnique', dest='qtechnique', default='fastnn',                            help='Query algorithm:fastnn|pquant|lsh')
    parser.add_argument('-query',      dest='query',                        action='store_true',       help='Query boolean (perform query or train')
    parser.add_argument('-parallel',   dest='parallel',                     action='store_true',       help='Parallel flag')
    parser.add_argument('-showfig',    dest='showfig',                      action='store_true',       help='Show fig')
    

    print 'Parsing arguments....'
    if len(sys.argv)>1:
        parameters = sys.argv[1]
    else:
        parameters = 'default.parameters.linux'
    text_file = open(parameters )
    args = parser.parse_args(text_file.read().split())
    text_file.close()
    print 'Arguments parsed!'

    #End Parser-------------------------------------------------------------------------------------------

    DatasetFolder = args.ds_folder
    OutputFolder = args.out_folder

    QueryDir  = DatasetFolder + 'query/'
    trainDir  = DatasetFolder + 'train/'
    bowDir    = OutputFolder + 'bow/'
    fisherDir = OutputFolder + 'fisher/'
    vladDir   = OutputFolder + 'vlad/'
    lshDir    = OutputFolder + 'lsh_'+args.global_des+'/'
    pqDir     = OutputFolder + 'pq_'+args.global_des+'/'

    if not os.path.isdir(OutputFolder): os.mkdir(OutputFolder)
    if not os.path.isdir(bowDir):       os.mkdir(bowDir)
    if not os.path.isdir(fisherDir):    os.mkdir(fisherDir)
    if not os.path.isdir(vladDir):      os.mkdir(vladDir)
    if not os.path.isdir(lshDir):       os.mkdir(lshDir)
    if not os.path.isdir(pqDir):        os.mkdir(pqDir)
    
    # library files
    libFiles_car = tl.getFilelist(trainDir+'car/','JPEG')[0]        #full path
    libFiles_dog = tl.getFilelist(trainDir+'dog/','JPEG')[0]
    LIBFILES = libFiles_car + libFiles_dog
    # query files
    QUERYFILES = tl.getFilelist(QueryDir+'dog/','JPEG')[0] + \
                 tl.getFilelist(QueryDir+'car/','JPEG')[0]

        
    # pq_m: number of splits in product quantization; 
    # pq_dist_mode: sdc (symmetric) / adc (asymmetric)
    # lsh: l = number of buckets/hashing tables used; maxbits = maximum number of bits per bucket
    
    params = {'patchSize':16, 'gridSpacing':16,\
              'keypointDetector':args.detector,\
              'descriptor':args.descriptor,\
              'K': args.K, 'nPt': args.npt, \
              'harris_sigma':3, 'harris_min_dist':10, 'harris_threshold':0.1, 'harris_wid':5,\
              'surf_extended':1,'surf_hessianThreshold':0.1, 'surf_nOctaveLayers':1, 'surf_nOctaves':1,\
              'lbp_patch_size': 50,\
              'classes':['car','dog'],\
              'global':args.global_des,\
              'pq_m': args.pqm,'pq_K': args.pqK, 'pq_dist_mode':'adc',\
              'pca': 60,\
              'lsh': args.lsh, 'lsh_l': 8, 'lsh_maxbits': 8}

    #
    #    FISHER VECTOR
    #    > any descriptor can be used, as in BoW
    #
    #    PRODUCT QUANTIZATION
    #    > needs GLOBAL image descriptors: GIST, CENTRIST, FISHER VECTOR, VLAD etc.
    #

    runmultiprocess = args.parallel

    #Replace-query
    query = args.query

    showFig = args.showfig
    compute_global = args.getglobal
    
    # 
    # query technique
    #
    bow = args.global_des == 'bow'
    fisher = args.global_des == 'fisher'
    vlad = args.global_des == 'vlad'
    product_quantization = False

    fastnn = False
    if args.qtechnique == 'fastnn':
        fastnn = True
    elif args.qtechnique == 'pquant':
        product_quantization = True

    #
    # check for incompatibility
    #
    
    if bow and params['keypointDetector'] == 'void':
        print 'ERROR: fisher vector is based on keypoints and local image descriptors!'
        exit(-1)
    if fisher and params['keypointDetector'] == 'void':
        print 'ERROR: fisher vector is based on keypoints and local image descriptors!'
        exit(-1)
    if vlad and params['keypointDetector'] == 'void':
        print 'ERROR: vlad vector is based on keypoints and local image descriptors!'
        exit(-1)
    start_time = time.time()
    if not query:
        #
        # PREPARE SEARCH LIBRARY
        #
    
        # compute fisher vector of the entire train dataset
        if compute_global and fisher:
            print('Compute fisher vector')
            fv.computeFisherVectors(LIBFILES,params,runmultiprocess,fisherDir,showFig)

        # compute vlad of the entire train dataset and store the codebook
        if compute_global and vlad:
            print('compute VLAD')
            vl.computeVLAD(LIBFILES,params,runmultiprocess,vladDir,showFig)

        # compute vlad of the entire train dataset and store the codebook
        if compute_global and bow:
            print('compute BOW')
            bw.computeBoW(LIBFILES,params,runmultiprocess,bowDir,args.showfig)
            
        # compute product quantization of the entire train dataset and store the codebook
        if product_quantization:
            print('computeProductQuantization')
            if bow:
                LIBRARY,UNUSED = tl.getFilelist(bowDir,'pkl','bw_')
            elif fisher:
                LIBRARY,UNUSED = tl.getFilelist(fisherDir,'pkl','fv_')
            elif vlad:
                LIBRARY,UNUSED = tl.getFilelist(vladDir,'pkl','vl_')
        
            pq.computeProductQuantization(LIBRARY,params,showFig,pqDir)
    
        # 2b. compute locality sensitive hashing on product quantization and store codes
        if product_quantization and params['lsh']:
            print('compute locality sensitive hashing')
            lsh.computeLSH(pqDir,params['pq_K'],params,lshDir)
    
    else:
        print('QUERY: image retrieval')
        #
        # QUERY: image retrieval
        #    
        
        MAP   = 0
        MAP5  = 0
        MAP10 = 0
        MAP15 = 0
        MAP20 = 0
        MAP25 = 0
    
        if bow:
            f = open(bowDir+'BOW_CODING_'+params['descriptor'][0]+'.pkl', 'rb')
            centroids = pickle.load(f)
            meanX = pickle.load(f)
            stdX = pickle.load(f)
            pca = pickle.load(f)
            f.close()
        elif fisher:
            f = open(fisherDir+'FISHER_GMM_'+params['descriptor'][0]+'.pkl', 'rb')
            clf = pickle.load(f)
            meanX = pickle.load(f)
            stdX = pickle.load(f)
            pca = pickle.load(f)
            f.close()                
        elif vlad:
            f = open(vladDir+'VLAD_CODING_'+params['descriptor'][0]+'.pkl', 'rb')
            centroids = pickle.load(f)
            meanX = pickle.load(f)
            stdX = pickle.load(f)
            pca = pickle.load(f)
            f.close()

        if fastnn:
            
            if bow:
                LIBRARY,UNUSED = tl.getFilelist(bowDir,'pkl','bw_')
            elif fisher:
                LIBRARY,UNUSED = tl.getFilelist(fisherDir,'pkl','fv_')
            elif vlad:
                LIBRARY,UNUSED = tl.getFilelist(vladDir,'pkl','vl_')
         
        
            trainDescriptors = [None]*len(LIBRARY)
            for nfile in range(0,len(LIBRARY)):
                print LIBRARY[nfile]
                f = open(LIBRARY[nfile], 'rb')
                trainDescriptors[nfile] = pickle.load(f)
                f.close()
            trainDescriptors=np.squeeze(np.array(trainDescriptors))
            
            aux=0
            for filename in QUERYFILES:
                aux +=1
                print "----> " + str(aux) + "/" + str(len(QUERYFILES)) + "  " + filename
                img = cv2.imread(filename)
                if bow:
                    testDescriptor = bw.bowVector(img,centroids,params, meanX, stdX, pca)
                if fisher:
                    testDescriptor = fv.FisherVector(img, params, clf, meanX, stdX, pca)               
                elif vlad:
                    testDescriptor = vl.vladVector(img,centroids,params, meanX, stdX, pca)

                # FAST-NN
                idxs= tl.fastnn(testDescriptor, trainDescriptors,k=25)                   
                                
                #print idxs

                MAP   += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 1)
                MAP5  += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 5)
                MAP10 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 10)
                MAP15 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 15)
                MAP20 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 20)
                MAP25 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 25)

                # show result
                if showFig: 
                    tl.showQueryResult(filename,trainDir,LIBRARY,idxs,params['global'])
                    raw_input("Press enter to continue")
    
        #
        #    PRODUCT QUANTIZATION
        #    
        if product_quantization:
            if params['lsh']:
                LIBRARY = tl.getFilelist(lshDir,'pkl','ls_')[0]
            else:
                LIBRARY = tl.getFilelist(pqDir,'pkl','pq_')[0]
                        
            # load codebook
            f = open(pqDir+'PQ_codebook_'+params['global']+'.pkl', 'rb')
            CODEBOOK =     pickle.load(f)
            f.close()
            
            tcount = 0
                                      
            aux = 0
            for filename in QUERYFILES:
                aux +=1
                print "----> " + str(aux) + "/" + str(len(QUERYFILES)) + "  " + filename
                img = cv2.imread(filename)
                if bow:
                    x = bw.bowVector(img,centroids,params, meanX, stdX, pca)
                elif fisher:
                    x = fv.FisherVector(img, params, clf, meanX, stdX, pca)               
                elif vlad:
                    x = vl.vladVector(img,centroids,params, meanX, stdX, pca)
                            
                # compute query results: idxs are the indices of (sorted) most similar images
                ts = time.time()         
                
                idxs = pq.pq_query(x,CODEBOOK,params,LIBRARY,lshDir)        
                
                tcount += time.time()-ts
                # show result
                    
                MAP   += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 1, args)
                MAP5  += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 5, args)
                MAP10 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 10, args)
                MAP15 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 15, args)
                MAP20 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 20, args)
                MAP25 += tl.MeanAveragePrecision(filename, trainDir, LIBRARY, idxs, 25, args)
                if showFig: 
                    tl.showQueryResult(filename,trainDir,LIBRARY,idxs,'PRODUCT QUANTIZATION')
                    raw_input("Press enter to continue")

            print 'Query time for PQ per image: %2.2f sec' % (tcount/len(QUERYFILES))


        mean_aver = (MAP/len(QUERYFILES))
        mean_aver5 = (MAP5/len(QUERYFILES))
        mean_aver10 = (MAP10/len(QUERYFILES))
        mean_aver15 = (MAP15/len(QUERYFILES))
        mean_aver20 = (MAP20/len(QUERYFILES))
        mean_aver25 = (MAP25/len(QUERYFILES))

        print 'performance   : %2.2f ' % mean_aver
        print 'performance  5: %2.2f ' % mean_aver5
        print 'performance 10: %2.2f ' % mean_aver10
        print 'performance 25: %2.2f ' % mean_aver25


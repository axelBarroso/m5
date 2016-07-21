from BOVW_functions import *
import configuration as cfg


def SVM(DSC_train, KPTS_train, DSC_test, KPTS_test, GT_ids_train, GT_ids_test):
    c = 1

    print 'Computing '+str(cfg.K)+' codebook'
    CB=getAndSaveCodebook(DSC_train, cfg.num_samples, cfg.K, cfg.codebook_filename)
    # CB=cPickle.load(open(codebook_filename,'r'))

    if cfg.spatialPyramid:
        print 'Computing '+str(cfg.K)+' VW_train'
        filenames_train,GT_ids_train,GT_labels_train = prepareFiles(cfg.path_train)
        VW_train=getAndSaveBoVW_SPMRepresentation(DSC_train,KPTS_train,cfg.K,CB,cfg.visual_words_SPM_filename_train,filenames_train,cfg.SPConfiguration)

        print 'Computing '+str(cfg.K)+' VW_test'
        filenames_test,GT_ids_test,GT_labels_test = prepareFiles(cfg.path_test)
        VW_test=getAndSaveBoVW_SPMRepresentation(DSC_test,KPTS_test,cfg.K,CB,cfg.visual_words_SPM_filename_test,filenames_test,cfg.SPConfiguration)

    else:
        print 'Computing '+str(cfg.K)+' VW_train'
        VW_train=getAndSaveBoVWRepresentation(DSC_train,cfg.K,CB,cfg.visual_words_filename_train)

        print 'Computing '+str(cfg.K)+' VW_test'
        VW_test=getAndSaveBoVWRepresentation(DSC_test,cfg.K,CB, cfg.visual_words_filename_test)

    print 'Computing '+str(cfg.K)+' SVM'

    if cfg.isCrossValidation:
        if cfg.kernel == 'rbf':
            accuracy, _, prob_train, prob_test = trainAndTestRBFSVM_withfolds(VW_train, VW_test,GT_ids_train,GT_ids_test, cfg.folds)
        elif cfg.kernel == 'linear':
            accuracy, _, prob_train, prob_test = trainAndTestLinearSVM_withfolds(VW_train, VW_test, GT_ids_train, GT_ids_test, cfg.folds)
        elif cfg.kernel == 'chi2':
            accuracy, _, prob_train, prob_test = trainAndTestChi2SVM_withfolds(VW_train, VW_test, GT_ids_train, GT_ids_test, cfg.folds)
        elif cfg.kernel == 'spm':
            accuracy = trainAndTestSPMSVM_withfolds(VW_train, VW_test, GT_ids_train, GT_ids_test,cfg.K, cfg.folds)

    else:
        if cfg.kernel == 'linear':
            accuracy, prediction, decision_function, _, _ = trainAndTestLinearSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,c)
        elif cfg.kernel == 'rbf':
            accuracy, prediction, decision_function, _, _ = trainAndTestRBFSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,c)
        elif cfg.kernel == 'chi2':
            accuracy, prediction, decision_function, _, _ = trainAndTestChi2SVM(VW_train,VW_test,GT_ids_train,GT_ids_test,c)
        elif cfg.kernel == 'spm':
            accuracy = trainAndTestSPMSVM(VW_train, VW_test, GT_ids_train, GT_ids_test, c, cfg.K)

    print 'Accuracy k='+str(cfg.K) + ' --> ' + str(accuracy)
    return [accuracy, prediction, decision_function]


def SVMLateFusionValidation(DSC_train, CDSC_train, DSC_test, CDSC_test, GT_ids_train, GT_ids_test):
    c = 1

    print 'Computing '+str(cfg.K)+' codebook'
    CB=getAndSaveCodebook(DSC_train, cfg.num_samples, cfg.K, cfg.codebook_filename)
    CB_Color=getAndSaveCodebook(CDSC_train, cfg.num_samples, cfg.K, cfg.codebook_Color_filename)
    # CB=cPickle.load(open(codebook_filename,'r'))

    print 'Computing '+str(cfg.K)+' VW_train'
    VW_train=getAndSaveBoVWRepresentation(DSC_train,cfg.K,CB,cfg.visual_words_filename_train)
    VW_Color_train=getAndSaveBoVWRepresentation(CDSC_train,cfg.K,CB_Color,cfg.visual_words_Color_filename_train)
    # VW_train=cPickle.load(open(visual_words_filename_train,'r'))

    print 'Computing '+str(cfg.K)+' VW_test'
    VW_test=getAndSaveBoVWRepresentation(DSC_test,cfg.K,CB,cfg.visual_words_filename_test)
    VW_Color_test=getAndSaveBoVWRepresentation(CDSC_test,cfg.K,CB_Color, cfg.visual_words_Color_filename_test)


    print 'Computing '+str(cfg.K)+' SVM'

    if cfg.isCrossValidation:
        if cfg.kernel == 'rbf':
            accuracy, _, prob_train, prob_test = trainAndTestRBFSVM_withfolds(VW_train, VW_test,GT_ids_train,GT_ids_test,cfg.folds)
            accuracyC, _, prob_Color_train, prob_Color_test = trainAndTestRBFSVM_withfolds(VW_Color_train,VW_Color_test,GT_ids_train,GT_ids_test,cfg.folds)
        elif cfg.kernel == 'linear':
            accuracy, _, prob_train, prob_test = trainAndTestLinearSVM_withfolds(VW_train, VW_test, GT_ids_train, GT_ids_test, cfg.folds)
            accuracyC, _, prob_Color_train, prob_Color_test = trainAndTestRBFSVM_withfolds(VW_Color_train, VW_Color_test,GT_ids_train,GT_ids_test,cfg.folds)
        elif cfg.kernel == 'chi2':
            accuracy, _, prob_train, prob_test = trainAndTestChi2SVM_withfolds(VW_train, VW_test, GT_ids_train, GT_ids_test, cfg.folds)
            accuracyC, _, prob_Color_train, prob_Color_test = trainAndTestRBFSVM_withfolds(VW_Color_train, VW_Color_test,GT_ids_train,GT_ids_test,cfg.folds)
    else:
        if cfg.kernel == 'rbf':
            accuracy, _, _, prob_train, prob_test = trainAndTestRBFSVM(VW_train, VW_test,GT_ids_train,GT_ids_test,c)
            accuracyC, _, _, prob_Color_train, prob_Color_test = trainAndTestRBFSVM(VW_Color_train,VW_Color_test,GT_ids_train,GT_ids_test,c)
        elif cfg.kernel == 'linear':
            accuracy, _, _, prob_train, prob_test = trainAndTestLinearSVM(VW_train,VW_test, GT_ids_train, GT_ids_test, c)
            accuracyC, _, _, prob_Color_train, prob_Color_test = trainAndTestLinearSVM(VW_Color_train,VW_Color_test,GT_ids_train,GT_ids_test,c)
        elif cfg.kernel == 'chi2':
            accuracy, _, _, prob_train, prob_test = trainAndTestChi2SVM(VW_train,VW_test, GT_ids_train, GT_ids_test, c)
            accuracyC, _, _, prob_Color_train, prob_Color_test = trainAndTestChi2SVM(VW_Color_train,VW_Color_test,GT_ids_train,GT_ids_test,c)

    prob_train_max        =  prob_train.max()
    prob_Color_train_max  =  prob_Color_train.max()
    prob_test_max         =  prob_test.max()
    prob_Color_test_max   =  prob_Color_test.max()

    bestAccuracy = 0
    bestBeta = 0
    betaArray = np.linspace(0.1, 0.9, 9)
    for beta in betaArray:
        train = beta*prob_train/prob_train_max
        trainColor = (1-beta)*prob_Color_train/prob_Color_train_max
        test = prob_test/prob_test_max
        testColor = prob_Color_test/prob_Color_test_max
        late_train = np.hstack((train,trainColor))
        late_test =  np.hstack((test,testColor))
        stdSlr = StandardScaler().fit(late_train)
        late_train_scaled = stdSlr.transform(late_train)
        late_test_scaled =  stdSlr.transform(late_test)
        clf_LATE = svm.SVC(kernel='linear', C=1).fit(late_train_scaled, GT_ids_train)
        ac_BOVW_LF = 100*clf_LATE.score(late_test_scaled,GT_ids_test)
        print 'Accuracy for beta: '+str(beta)+' Gives us Accuracy of :' + str(ac_BOVW_LF)
        if ac_BOVW_LF > bestAccuracy:
            bestAccuracy = ac_BOVW_LF
            bestBeta = beta


    print 'Accuracy BOVW Descriptor: '+str(accuracy)
    print 'Accuracy BOVW Color descriptor: '+str(accuracyC)
    print 'Accuracy BOVW Late Fusion: '+str(bestAccuracy)
    return bestAccuracy


def SVMIntermediateFusionValidation(DSC_train, CDSC_train, KPTS_train, DSC_test, CDSC_test, KPTS_test, GT_ids_train, GT_ids_test):
    c = 1

    print 'Computing '+str(cfg.K)+' codebook'
    CB=getAndSaveCodebook(DSC_train, cfg.num_samples, cfg.K, cfg.codebook_filename)
    CB_Color=getAndSaveCodebook(CDSC_train, cfg.num_samples, cfg.K, cfg.codebook_Color_filename)

    if cfg.spatialPyramid:
        print 'Computing Pyramid'+str(cfg.K)+' VW_train'
        filenames_train,GT_ids_train,GT_labels_train = prepareFiles(cfg.path_train)
        VW_train=getAndSaveBoVW_SPMRepresentation(DSC_train,KPTS_train,cfg.K,CB,cfg.visual_words_SPM_filename_train,filenames_train,cfg.SPConfiguration)
        VW_Color_train=getAndSaveBoVW_SPMRepresentation(CDSC_train,KPTS_train,cfg.K,CB_Color,cfg.visual_words_SPM_Color_filename_train,filenames_train,cfg.SPConfiguration)

        print 'Computing Pyramid'+str(cfg.K)+' VW_test'
        filenames_test,GT_ids_test,GT_labels_test = prepareFiles(cfg.path_test)
        VW_test=getAndSaveBoVW_SPMRepresentation(DSC_test,KPTS_test,cfg.K,CB,cfg.visual_words_SPM_filename_test,filenames_test,cfg.SPConfiguration)
        VW_Color_test=getAndSaveBoVW_SPMRepresentation(CDSC_test,KPTS_test,cfg.K,CB_Color,cfg.visual_words_SPM_Color_filename_test,filenames_test,cfg.SPConfiguration)

    else:
        print 'Computing '+str(cfg.K)+' VW_train'
        VW_train=getAndSaveBoVWRepresentation(DSC_train,cfg.K,CB,cfg.visual_words_filename_train)
        VW_Color_train=getAndSaveBoVWRepresentation(CDSC_train,cfg.K,CB_Color,cfg.visual_words_Color_filename_train)

        print 'Computing '+str(cfg.K)+' VW_test'
        VW_test=getAndSaveBoVWRepresentation(DSC_test,cfg.K,CB,cfg.visual_words_filename_test)
        VW_Color_test=getAndSaveBoVWRepresentation(CDSC_test,cfg.K,CB_Color,cfg.visual_words_Color_filename_test)


    print 'Computing '+str(cfg.K)+' SVM'
    bestAccuracy = 0
    bestBeta = 0
    betaArray = np.linspace(0.1, 0.9, 9)
    for beta in betaArray:
        train = beta * VW_train
        trainC = (1-beta) * VW_Color_train
        VW_inter_train = np.hstack((train,trainC))
        VW_inter_test = np.hstack((VW_test,VW_Color_test))
        if cfg.isCrossValidation:
            if cfg.kernel == 'rbf':
                accuracy, _, prob_train, prob_test = trainAndTestRBFSVM_withfolds(VW_inter_train, VW_inter_test,GT_ids_train,GT_ids_test,cfg.folds)
            elif cfg.kernel == 'linear':
                accuracy, _, prob_train, prob_test = trainAndTestLinearSVM_withfolds(VW_inter_train, VW_inter_test, GT_ids_train, GT_ids_test, cfg.folds)
            elif cfg.kernel == 'chi2':
                accuracy, _, prob_train, prob_test = trainAndTestChi2SVM_withfolds(VW_inter_train, VW_inter_test, GT_ids_train, GT_ids_test, cfg.folds)
            elif cfg.kernel == 'spm':
                accuracy = trainAndTestSPMSVM_withfolds(VW_inter_train, VW_inter_test, GT_ids_train, GT_ids_test,cfg.K, cfg.folds)
            elif cfg.kernel == 'his':
                accuracy = trainAndTestHISVM_withfolds(VW_inter_train, VW_inter_test, GT_ids_train, GT_ids_test, cfg.folds)
        else:
            if cfg.kernel == 'rbf':
                accuracy, _, prob_train, prob_test = trainAndTestRBFSVM(VW_inter_train, VW_inter_test,GT_ids_train,GT_ids_test, c)
            elif cfg.kernel == 'linear':
                accuracy, _, _, prob_train, prob_test = trainAndTestLinearSVM(VW_inter_train,VW_inter_test, GT_ids_train, GT_ids_test, c)
            elif cfg.kernel == 'chi2':
                accuracy, _, _, prob_train, prob_test = trainAndTestChi2SVM(VW_inter_train,VW_inter_test, GT_ids_train, GT_ids_test, c)
            elif cfg.kernel == 'spm':
                accuracy = trainAndTestSPMSVM(VW_inter_train, VW_inter_test, GT_ids_train, GT_ids_test, c, cfg.K)
            elif cfg.kernel == 'his':
                accuracy = trainAndTestHISVM(VW_inter_train, VW_inter_test, GT_ids_train, GT_ids_test, c)

        print 'Accuracy for beta: '+str(beta)+' Gives us Accuracy of :' + str(accuracy)
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestBeta = beta

    print 'Accuracy BOVW Intermediate Fusion: '+str(bestAccuracy)
    return bestAccuracy
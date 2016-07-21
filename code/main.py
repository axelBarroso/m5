from BOVW_functions import *
from SVM_functions import *
from aux_functions import *
import configuration as cfg

from joblib import Parallel, delayed

def run():

    init=time.time()

    filenames_train,GT_ids_train,GT_labels_train = prepareFiles(cfg.path_train)
    filenames_test,GT_ids_test,GT_labels_test = prepareFiles(cfg.path_test)

    KPTS_train, DSC_train, filenames_train, GT_ids_train, GT_labels_train = getKeypointsDescriptors(filenames_train, cfg.detector, cfg.descriptor, cfg.color_constancy, GT_ids_train, GT_labels_train, cfg.num_samples)
    KPTS_test, DSC_test, filenames_test, GT_ids_test, GT_labels_test = getKeypointsDescriptors(filenames_test, cfg.detector, cfg.descriptor, cfg.color_constancy, GT_ids_test, GT_labels_test, cfg.num_samples)

    DSC_train, DSC_test = data_whitening(DSC_train,DSC_test)

    # COLOR DESCRIPTOR
    if cfg.fusion != 'none':
        if cfg.descriptorColor == 'Histogram':
            CDSC_test = getLocalColorDescriptors(filenames_test,KPTS_test, cfg.color_constancy)
            CDSC_train = getLocalColorDescriptors(filenames_train,KPTS_train, cfg.color_constancy)
        elif cfg.descriptorColor == 'PROB':
            CDSC_test = getLocalColorProbabilities(filenames_test,KPTS_test, cfg.color_constancy)
            CDSC_train = getLocalColorProbabilities(filenames_train,KPTS_train, cfg.color_constancy)

        CDSC_train, CDSC_test = data_whitening(CDSC_train,CDSC_test)

    if cfg.fusion == 'late':
        DSC_train, DSC_test = performPCA(DSC_train, DSC_test, cfg.isRandomized)
        CDSC_train, CDSC_test = performPCA(CDSC_train, CDSC_test, cfg.isRandomized)

        accuracy = SVMLateFusionValidation(DSC_train, CDSC_train, DSC_test, CDSC_test, GT_ids_train, GT_ids_test)

    elif cfg.fusion == 'intermediate':
        # DSC_train, DSC_test = performPCA(DSC_train, DSC_test, cfg.isRandomized)
        # CDSC_train, CDSC_test = performPCA(CDSC_train, CDSC_test, cfg.isRandomized)

        accuracy = SVMIntermediateFusionValidation(DSC_train, CDSC_train, KPTS_train, DSC_test, CDSC_test, KPTS_test, GT_ids_train, GT_ids_test)

    elif cfg.fusion == 'early':
        DSC_train, DSC_test = earlyFusion(DSC_train, CDSC_train, DSC_test, CDSC_test)
        DSC_train, DSC_test = performPCA(DSC_train, DSC_test, cfg.isRandomized)

        #results = Parallel(n_jobs=1)(delayed(SVM)(detector, descriptor, num_samples, FDSC_train, FDSC_test, GT_ids_train, GT_ids_test, k, kernel)for k in K)
        accuracy, predictionsList, decisionFunctionList = SVM( DSC_train, KPTS_train, DSC_test, KPTS_test,GT_ids_train, GT_ids_test)

    else:
        # DSC_train, DSC_test = performPCA(DSC_train, DSC_test, cfg.isRandomized)

        #results = Parallel(n_jobs=1)(delayed(SVM)(detector, descriptor, num_samples, FDSC_train, FDSC_test, GT_ids_train, GT_ids_test, k, kernel)for k in K)
        accuracy, predictionsList, decisionFunctionList = SVM(DSC_train, KPTS_train, DSC_test, KPTS_test, GT_ids_train, GT_ids_test)

    # Parallel!
    # for i in range(len(K)):
    #     accuracy[i]                =   results[i][0]
    #     predictionsList            =   results[i][1]
    #     decisionFunctionList[i]    =   results[i][2]

    # plotResults(K, accuracy, decisionFunctionList, predictionsList, GT_ids_train, GT_ids_test)

    end=time.time()

    print 'Total time (' + cfg.detector + ' - ' + cfg.descriptor + ' - Fusion: '+ cfg.fusion + ') : ' +str(end-init)+ ' secs.'
    print 'Evaluation Finished'



if __name__ == '__main__':
    run()
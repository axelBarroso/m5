from BOVW_functions import *


detector='Dense'
descriptor='SIFT'
num_samples=50000
k=1000
c = 1
folds = 5

#SPCOnfiguration: If is 1: I+2x2+4x4 grid. If is 2: I+3x1 grid. If is 3: I+3x3 grid. If is 4 I+2x2 grid.
SPConfiguration = 1

codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_SPM_filename_train='VWSPM_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_SPM_filename_test='VWSPM_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../Databases/MIT_split/train/')
KPTS_train, DSC_train, filenames_train, GT_ids_train, GT_labels_train = getKeypointsDescriptors(filenames_train,detector,descriptor, 0, GT_ids_train, GT_labels_train, num_samples)

#Compute the codebook
CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)

#Compute train visual words (normal and pyramid)
VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
VWSPM_train=getAndSaveBoVW_SPMRepresentation(DSC_train,KPTS_train,k,CB,visual_words_SPM_filename_train,filenames_train,SPConfiguration)


#Compute test visual words (normal and pyramid)
filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../Databases/MIT_split/test/')
KPTS_test, DSC_test, filenames_test, GT_ids_test, GT_labels_test = getKeypointsDescriptors(filenames_test,detector,descriptor, 0, GT_ids_test, GT_labels_test, num_samples)
VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)
VWSPM_test=getAndSaveBoVW_SPMRepresentation(DSC_test,KPTS_test,k,CB,visual_words_SPM_filename_test,filenames_test,SPConfiguration)

#Linear SVM with normal and pyramid
# ac_BOVW_L, prediction, decision_function, _, _ = trainAndTestLinearSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,c)
# ac_BOVW_SPM_L, prediction, decision_function, _, _ = trainAndTestLinearSVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,c)
# print 'Accuracy BOVW with LinearSVM: '+str(ac_BOVW_L)
# print 'Accuracy BOVW with SPM with LinearSVM:'+str(ac_BOVW_SPM_L)


#RBF SVM with normal and pyramid
# ac_BOVW_RBF, prediction, decision_function, _, _ = trainAndTestRBFSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,c)
# ac_BOVW_SPM_RBF, prediction, decision_function, _, _ = trainAndTestRBFSVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,c)
# print 'Accuracy BOVW with RBF: '+str(ac_BOVW_RBF)
# print 'Accuracy BOVW with SPM with RBF: '+str(ac_BOVW_SPM_RBF)
#
# #CCHI2 SVM with folds with normal and pyramid
# ac_BOVW_CHI2, prediction, _, _ = trainAndTestChi2SVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds)
# ac_BOVW_SPM_CHI2, prediction, _, _ = trainAndTestChi2SVM_withfolds(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,folds)
# print 'Accuracy BOVW with Chi2: '+str(ac_BOVW_CHI2)
# print 'Accuracy BOVW with SPM with Chi2: '+str(ac_BOVW_SPM_CHI2)
#
# #HISVM with normal and pyramid
# ac_BOVW_HI = trainAndTestHISVM(VW_train,VW_test,GT_ids_train,GT_ids_test,c)
# ac_BOVW_SPM_HI = trainAndTestHISVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,c)
# print 'Accuracy BOVW with HISVM: '+str(ac_BOVW_HI)
# print 'Accuracy BOVW with SPM with HISVM: '+str(ac_BOVW_SPM_HI)

#Spatial Pyramid SVM (kernel coded at BovW_functions) with pyramid
ac_BOVW_SPM_SPMK = trainAndTestSPMSVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,c,k)
print 'Accuracy BOVW with SPM with SPMKernelSVM: '+str(ac_BOVW_SPM_SPMK)

#Spatial Pyramid SVM with folds(kernel coded at BovW_functions) with pyramid
ac_BOVW_SPM_SPMK_folds = trainAndTestSPMSVM_withfolds(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test, k, folds)
print 'Accuracy BOVW with SPM with SPMKernelSVM with folds: '+str(ac_BOVW_SPM_SPMK_folds)
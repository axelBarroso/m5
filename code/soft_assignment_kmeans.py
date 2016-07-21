from BOVW_functions import *
from sklearn.cluster import KMeans


detector='Dense'
descriptor='SIFT'
num_samples=50000
R = 10 #Num centroid to consider
alpha = 50 #Distance penalization

k=1000
C=1

codebook_filename='codebook_soft.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../Databases/MIT_split/train/')
KPTS_train, DSC_train, filenames_train, GT_ids_train, GT_labels_train = getKeypointsDescriptors(filenames_train,detector,descriptor, 0, GT_ids_train, GT_labels_train, num_samples)
CB=getAndSaveCodebookSoftAsignementKMeans(DSC_train, k, codebook_filename)
# CB=cPickle.load(open(codebook_filename,'rb'))

VW_train=getAndSaveBoVWRepresentationSoftAssignementKMeans(DSC_train, k, CB, R, alpha)

filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../Databases/MIT_split/test/')
KPTS_test, DSC_test, filenames_test, GT_ids_test, GT_labels_test = getKeypointsDescriptors(filenames_test,detector,descriptor, 0, GT_ids_test, GT_labels_test, num_samples)
VW_test=getAndSaveBoVWRepresentationSoftAssignementKMeans(DSC_test, k, CB, R, alpha)

ac_BOVW_RBF, predictions, decision_function, _, _ = trainAndTestRBFSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,C)
print 'Accuracy BOVW RBF: '+str(ac_BOVW_RBF)

ac_BOVW_CHI2, prediction, _, _ = trainAndTestChi2SVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,5)
print 'Accuracy BOVW CHI: '+str(ac_BOVW_CHI2)

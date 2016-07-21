from BOVW_functions import *


detector='SIFT'
descriptor='SIFT'																																															
num_samples=50000

k=32
folds=5
start=0.01
end=10
numparams=30


codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../Databases/MIT_split/train/')
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)
#CB=cPickle.load(open(codebook_filename,'r'))

VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
#VW_train=cPickle.load(open(visual_words_filename_train,'r'))


filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../Databases/MIT_split/test/')
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)

ac_BOVW_L, SVMpredictions = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

print 'Accuracy BOVW: '+str(ac_BOVW_L)


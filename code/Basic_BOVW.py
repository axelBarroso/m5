from BOVW_functions import *
import Evaluate_Results
import matplotlib.pyplot as plt


detector='SIFT'
descriptor='SIFT'																																															
num_samples=50000

k=32
# Cs=range(1,3,1)
Cs = [0.001, 0.0051, 0.01, 0.051, 0.1, 0.51, 1, 5.1, 10, 51, 100]
codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../Databases/MIT_split/train/')
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)
CB=cPickle.load(open(codebook_filename,'r'))

VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
VW_train=cPickle.load(open(visual_words_filename_train,'r'))


filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../Databases/MIT_split/test/')
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)

# Compute labels for each C
predictionsList = np.zeros(shape=(len(Cs), len(GT_ids_test)))
accuracy = np.zeros(len(Cs))

i = 0
for C in Cs:
    ac_BOVW_L, predictions = trainAndTestLinearSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,C)
    #ac_BOVW_RBF = trainAndTestRBFSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,C)
	#ac_BOVW_POLY = trainAndTestPolySVM(VW_train,VW_test,GT_ids_train,GT_ids_test,C)
	#ac_BOVW_SIG = trainAndTestSigSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,C)
    print 'Accuracy BOVW: '+str(ac_BOVW_L)
    predictionsList[i,:] = predictions
    accuracy[i] = ac_BOVW_L
    i += 1

#Plot accuracy
plt.figure()
# Plot class ROCS
plt.plot(Cs, accuracy)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C')
plt.show()

# Compute ROC (for all C (or choosen parameter))
parameter = Cs
Evaluate_Results.computeROC(GT_ids_test, predictionsList, parameter)

#Compute confusion matrix for one C
Evaluate_Results.confusionMatrix(GT_ids_test, predictionsList[0,:])

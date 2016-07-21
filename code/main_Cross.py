from BOVW_functions import *
import Evaluate_Results
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def run():

    detector='SIFT'
    descriptor='SIFT'
    num_samples=50000
    color_constancy = 0 #If is 1 is applied

    K = range(900, 1300, 108)
    filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../Databases/MIT_split_min/train/')
    filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../Databases/MIT_split_min/test/')

    KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor, color_constancy)
    KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor, color_constancy)

    predictionsList = np.zeros(shape=(len(K), len(GT_ids_test)))
    accuracy = np.zeros(len(K))

    results = Parallel(n_jobs=3)(delayed(SVMcrossValidation)(k, detector, descriptor, num_samples, DSC_train, DSC_test, GT_ids_train, GT_ids_test)for k in K)


    for i in range(len(results)):
        accuracy[i] = results[i][0]
        predictionsList[i] = results[i][1]


    #Plot accuracy
    plt.figure(1)
    # Plot class ROCS
    plt.plot(K, accuracy)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Vocabulary Size')
    # plt.show()

    # Compute ROC (for all Vocabulary Size (or choosen parameter))
    Evaluate_Results.computeROC(GT_ids_test, predictionsList, K)

    #Compute confusion matrix for one Vocabulary Size
    # Evaluate_Results.confusionMatrix(GT_ids_test, predictionsList[0,:])

    print 'Evaluation Finished'


def SVMcrossValidation(k, detector, descriptor, num_samples, DSC_train, DSC_test, GT_ids_train, GT_ids_test):

    folds = 9

    codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
    visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
    visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

    print 'Computing '+str(k)+' codebook'
    CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)
    # CB=cPickle.load(open(codebook_filename,'r'))

    print 'Computing '+str(k)+' VW_train'
    VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
    # VW_train=cPickle.load(open(visual_words_filename_train,'r'))

    print 'Computing '+str(k)+' VW_test'
    VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)

    print 'Computing '+str(k)+' SVM'
    accuracy, SVMpredictions = trainAndTestChi2SVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds)

    print 'Accuracy k='+str(k) + ' --> ' + str(accuracy)

    return [accuracy, SVMpredictions]

if __name__ == '__main__':
    run()
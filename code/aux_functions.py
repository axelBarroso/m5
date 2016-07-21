import Evaluate_Results
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np


def plotResults(K, accuracy, decisionFunctionList, predictionsList, GT_ids_train, GT_ids_test):

    # indexMax = np.argmax(accuracy)
    # Plot accuracy
    plt.figure(1)
    # Plot class ROCS
    plt.plot(K, accuracy)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Vocabulary Size')
    plt.show()

    # Compute ROC (for all Vocabulary Size (or choosen parameter))
    # Evaluate_Results.computeROC(GT_ids_test, decisionFunctionList[indexMax])
    Evaluate_Results.computeROC(GT_ids_test, decisionFunctionList[0])

    # Compute confusion matrix for one Vocabulary Size
    # Evaluate_Results.confusionMatrix(GT_ids_test, predictionsList[0,:])


def YCRCBhistogramEqualization(img):

    #convert img to YCR_CB
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    #equalize Y component (lighting)
    img2[:, :, 0] = cv2.equalizeHist(img2[:, :, 0])
    #convert this output image to rgb
    rgb = cv2.cvtColor(img2,cv2.COLOR_YCR_CB2BGR)

    return rgb


def earlyFusion(DSC_train, CDSC_train, DSC_test, CDSC_test):

    FDSC_test=[]
    FDSC_train=[]

    #Get the max to normalize descriptors [0-1]
    color_max = []
    shape_max = []
    color_min = []
    shape_min = []
    for i in range(len(CDSC_test)):
        color_max.append(CDSC_test[i].max())
        color_min.append(CDSC_test[i].min())
        shape_max.append(DSC_test[i].max())
        shape_min.append(DSC_test[i].min())

    color_max = np.amax(color_max)
    color_min = np.amin(color_min)
    shape_max = np.amax(shape_max)
    shape_min = np.amin(shape_min)
    color_norm = max(color_max, abs(color_min))
    shape_norm = max(shape_max, abs(shape_min))

    for i in range(len(DSC_test)):
       FDSC_test.append(np.hstack((DSC_test[i] / shape_norm, CDSC_test[i] / color_norm)))

    #Get the max to normalize descriptors [0-1]
    color_max = []
    shape_max = []
    color_min = []
    shape_min = []
    for i in range(len(DSC_train)):
        color_max.append(CDSC_train[i].max())
        color_min.append(CDSC_train[i].min())
        shape_max.append(DSC_train[i].max())
        shape_min.append(DSC_train[i].min())

    color_max = np.amax(color_max)
    color_min = np.amin(color_min)
    shape_max = np.amax(shape_max)
    shape_min = np.amin(shape_min)
    color_norm = max(color_max, abs(color_min))
    shape_norm = max(shape_max, abs(shape_min))

    for i in range(len(DSC_train)):
       FDSC_train.append(np.hstack((DSC_train[i] / shape_norm, CDSC_train[i] / color_norm)))

    return FDSC_train, FDSC_test
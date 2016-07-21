import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def computeROC(gt, predictions):
    #Computes and plot Confusion Matrix and ROC

    print 'Start evaluating results'
    num_classes = max(gt) + 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for c in range(num_classes):
        fpr[c], tpr[c],_ = roc_curve(gt, predictions[:,c], pos_label=c)
        roc_auc[c] = auc(fpr[c], tpr[c])
    #Inicialitations
    #predictions = predictions.tolist()


    # FP = np.zeros(shape=(len(parameter),num_classes))
    # TN = np.zeros(shape=(len(parameter),num_classes))
    # TP = np.zeros(shape=(len(parameter),num_classes))
    # FN = np.zeros(shape=(len(parameter),num_classes))
    # TPR = np.zeros(shape=(len(parameter),num_classes))
    # FPR = np.zeros(shape=(len(parameter),num_classes))
    #
    # meanTPR = np.zeros(len(parameter))
    # meanFPR = np.zeros(len(parameter))
    #
    # # Compute ROC
    # for p in range(len(parameter)):                 #For each parameter value
    #     for c in range(num_classes):    #For each class
    #         for i in range(len(gt)):    #For each image
    #             if int(predictions[p,i]) is gt[i]:
    #                 TP[p,c] += 1
    #             elif gt[i] is c:
    #                 FN[p,c] += 1
    #             elif int(predictions[p,i]) is c:
    #                 FP[p,c] += 1
    #             else:
    #                 TN[p,c] += 1
    #         TPR[p,c] = TP[p,c] / (TP[p,c] + FN[p,c])
    #         FPR[p,c] = 1 - (TN[p,c] / (TN[p,c] + FP[p,c]))
    #         meanTPR[p] += TPR[p,c]
    #         meanFPR[p] += FPR[p,c]
    #     meanTPR[p] /= num_classes
    #     meanFPR[p] /= num_classes

    colors = ['r', 'b', 'g', 'y', 'm', 'c', '#FE642E','#58FA82']
    #Plot ROC
    # plt.figure(2)
    plt.figure(1)
    # Plot class ROCS
    for c in range(num_classes):
        plt.plot(fpr[c],tpr[c], colors[c], label='ROC curve (area = %0.2f)' % roc_auc[c])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC')
    plt.legend()
    plt.show()


    # Plot mean ROC
    # plt.figure(3)
    # plt.plot(meanFPR,meanTPR, 'k-')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Mean ROC')
    # plt.show()


def confusionMatrix(gt,predictions):
    #Compute confusion matrix
    num_classes = max(gt) + 1
    confusionMatrix = np.zeros(shape=(num_classes,num_classes))
    for i in range(len(gt)):
        confusionMatrix[int(predictions[i])][int(gt[i])] += 1
    print "Confusion Matrix"
    print confusionMatrix

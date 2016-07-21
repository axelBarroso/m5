from BOVW_functions import *

##########################
#PARaMETROS DE EJECUCION
##########################

# Metodo usado para detectar los puntos de interes
detector='SIFT' #DENSE
# Metodo usado para obtener la descripcion de los puntos de interes
descriptor='SIFT'
# Nemero de puntos de interes que se utilizan para obtener el vocabulario visual
num_samples=50000
# Nemero de palabras en el vocabulario visual
k=1116
# Factor de regularizacion C para entrenar el clasificador SVM
C=1

color_constancy = 0 #If is 1 is applied
# Directorio raiz donde se encuentran todas las imagenes de aprendizaje
dataset_folder_train='../Databases/MIT_split_min/train/'
# Directorio raiz donde se encuentran todas las imagenes de test
dataset_folder_test='../Databases/MIT_split_min/test/'

##############################################


# Preparacion de los nombres de los ficheros necesarios para guardar el vocabulario y las palabras visuales de las imagenes de aprendizaje y test
# Se generan tres vocabularios y, por lo tanto tres conjuntos de palabras visuales utilizando tres configuraciones diferentes de descriptor:
# 1. Solo SIFT, 2. Color, 3. Concatenacion de SIFT + color

codebook_filename_SIFT='CB_S_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
codebook_filename_COLOR='CB_C_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
codebook_filename_EARLY='CB_E_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_train_SIFT='VW_train_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train_COLOR='VW_train_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train_EARLY='VW_train_EARLY_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_test_SIFT='VW_test_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test_COLOR='VW_test_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test_EARLY='VW_test_EARLY_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

# Calculo de puntos de interes para todas las imagenes del conjunto de aprendizaje. La descripcion se obtiene tanto con SIFT como con el descriptor de color
filenames_train,GT_ids_train,GT_labels_train = prepareFiles(dataset_folder_train)
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor,color_constancy)
CDSC_train = getLocalColorDescriptors(filenames_train,KPTS_train,color_constancy)

#Get the max to normalize descriptors [0-1]
color_max = np.amax(CDSC_train[:])
shape_max = np.amax(DSC_train[:])

# Early fusion de las dos descripciones para cada punto de interes: SIFT + color. Simplemente se concatenan los dos descriptores
FDSC_train=[]
for i in range(len(DSC_train)):
	FDSC_train.append(np.hstack((DSC_train[i] / shape_max ,CDSC_train[i] / color_max)))

# Construccion de los 3 vocabularios visuales: SIFT, Color y Early fusion. Los vocabularios quedan guardados en disco.
# Comentar estas lineas si los vocabularios ya estan creados y guardados en disco de una ejecucion anterior
CB_SIFT=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename_SIFT)
CB_COLOR=getAndSaveCodebook(CDSC_train, num_samples, k, codebook_filename_COLOR)
CB_EARLY=getAndSaveCodebook(FDSC_train, num_samples, k, codebook_filename_EARLY)

# Carga de los vocabularios visuales previamente creados y guardados en disco en una ejecucion anterior.
# Comentar estas lineas si se quiere re-calcular los vocabularios o si los vocabularios todavia no se han creado
#CB_SIFT=cPickle.load(open(codebook_filename_SIFT,'r'))
#CB_COLOR=cPickle.load(open(codebook_filename_COLOR,'r'))
#CB_EARLY=cPickle.load(open(codebook_filename_EARLY,'r'))

# Obtiene la descripcion BoW de las imagenes del conjunto de aprendizaje para las tres descripciones: SIFT, Color y Early fusion
VW_SIFT_train=getAndSaveBoVWRepresentation(DSC_train,k,CB_SIFT,visual_words_filename_train_SIFT)
VW_COLOR_train=getAndSaveBoVWRepresentation(CDSC_train,k,CB_COLOR,visual_words_filename_train_COLOR)
VW_FUSION_train=getAndSaveBoVWRepresentation(FDSC_train,k,CB_EARLY,visual_words_filename_train_EARLY)

# Carga de las 3 descripciones BoW del conjunto de aprendizaje previamente creadas y guardadas en disco en una ejecucion anterior.
# Comentar estas lineas si se quiere re-calcular la representacion o si la representacion todavia no se ha creado
#VW_SIFT_train=cPickle.load(open(visual_words_filename_train_SIFT,'r'))
#VW_COLOR_train=cPickle.load(open(visual_words_filename_train_COLOR,'r'))
#VW_FUSION_train=cPickle.load(open(visual_words_filename_train_EARLY,'r'))

# Calculo de puntos de interes para todas las imagenes del conjunto de test. Obtiene las dos descripciones (SIFT y color) y tambien las concatena
# para obtener la descripcion Early Fusion
filenames_test,GT_ids_test,GT_labels_test = prepareFiles(dataset_folder_test)
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor,color_constancy)
CDSC_test = getLocalColorDescriptors(filenames_test,KPTS_test,color_constancy)
FDSC_test=[]


#Get the max to normalize descriptors [0-1]
color_max = np.amax(CDSC_test[:])
shape_max = np.amax(DSC_test[:])

for i in range(len(DSC_test)):
	FDSC_test.append(np.hstack((DSC_test[i] / shape_max, CDSC_test[i] / color_max)))

# Obtiene las 3 descripciones BoW (SIFT, color y early fusion) de las imagenes del conjunto de test
VW_SIFT_test=getAndSaveBoVWRepresentation(DSC_test,k,CB_SIFT,visual_words_filename_test_SIFT)
VW_COLOR_test=getAndSaveBoVWRepresentation(CDSC_test,k,CB_COLOR,visual_words_filename_test_COLOR)
VW_FUSION_test=getAndSaveBoVWRepresentation(FDSC_test,k,CB_EARLY,visual_words_filename_test_EARLY)

# Entrena un clasificador SVM con las imagenes del conjunto de aprendizaje y lo evalua utilizando las imagenes del conjunto de test
# para las 3 descripciones (SIFT, color y early fusion)
# Devuelve la accuracy como medida del rendimiento del clasificador
ac_BOVW_SIFT = trainAndTestRBFSVM(VW_SIFT_train,VW_SIFT_test,GT_ids_train,GT_ids_test,C)
ac_BOVW_COLOR = trainAndTestRBFSVM(VW_COLOR_train,VW_COLOR_test,GT_ids_train,GT_ids_test,C)
ac_BOVW_EF = trainAndTestRBFSVM(VW_FUSION_train,VW_FUSION_test,GT_ids_train,GT_ids_test,C)


print 'Accuracy BOVW with LinearSVM SIFT: '+str(ac_BOVW_SIFT)
print 'Accuracy BOVW with LinearSVM Color: '+str(ac_BOVW_COLOR)
print 'Accuracy BOVW with LinearSVM Early Fusion: '+str(ac_BOVW_EF)
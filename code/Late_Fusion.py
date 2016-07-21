from BOVW_functions import *

##########################
#PARAMETROS DE EJECUCION
##########################

# Metodo usado para detectar los puntos de interes
detector='FAST'
# Metodo usado para obtener la descripcion de los puntos de interes
descriptor='SURF'
# Numero de puntos de interes que se utilizan para obtener el vocabulario visual
num_samples=50000
# Numero de palabras en el vocabulario visual
k=1000
# Factor de regularizacion C para entrenar el clasificador SVM
C=1
# Directorio raiz donde se encuentran todas las imagenes de aprendizaje
dataset_folder_train='../Databases/MIT_split_min/train/'
# Directorio raiz donde se encuentran todas las imagenes de test
dataset_folder_test='../Databases/MIT_split_min/test/'

##############################################


# Preparacion de los nombres de los ficheros necesarios para guardar el vocabulario y las palabras visuales de las imagenes de aprendizaje y test
# Se generan dos vocabularios y, por lo tanto dos conjuntos de palabras visuales utilizando dos configuraciones diferentes de descriptor:
# 1. SIlo SIFT, 2. Color
codebook_filename_SIFT='CB_S_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
codebook_filename_COLOR='CB_C_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_train_SIFT='VW_train_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train_COLOR='VW_train_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_test_SIFT='VW_test_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test_COLOR='VW_test_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'


# Calculo de puntos de interes para todas las imagenes del conjunto de aprendizaje. La descripcion se obtiene tanto con SIFT como con el descriptor de color
filenames_train,GT_ids_train,GT_labels_train = prepareFiles(dataset_folder_train)
KPTS_train,DSC_train, filenames_train, GT_ids_train, GT_labels_train = getKeypointsDescriptors(filenames_train,detector,descriptor, 0, GT_ids_train, GT_labels_train)
CDSC_train = getLocalColorDescriptors(filenames_train,KPTS_train, 0)

# Calculo de puntos de interes para todas las imagenes del conjunto de test. Obtiene las dos descripciones (SIFT y color)
filenames_test,GT_ids_test,GT_labels_test = prepareFiles(dataset_folder_test)
KPTS_test,DSC_test, filenames_test, GT_ids_test, GT_labels_test = getKeypointsDescriptors(filenames_test,detector,descriptor, 0, GT_ids_test, GT_labels_test)
CDSC_test = getLocalColorDescriptors(filenames_test,KPTS_test, 0)

DSC_PCA_SIFT_train, DSC_PCA_SIFT_test = performPCA(DSC_train, DSC_test, 0)
CDSC_PCA_COLOR_train, CDSC_PCA_COLOR_test = performPCA(CDSC_train, CDSC_test, 0)

# Construccion de los 2 vocabularios visuales: SIFT, Color. Los vocabularios quedan guardados en disco.
# Comentar estas lineas si los vocabularios ya estan creados y guardados en disco de una ejecucion anterior
CB_SIFT=getAndSaveCodebook(DSC_PCA_SIFT_train, num_samples, k, codebook_filename_SIFT)
CB_COLOR=getAndSaveCodebook(CDSC_PCA_COLOR_train, num_samples, k, codebook_filename_COLOR)

# Carga de los vocabularios visuales previamente creados y guardados en disco en una ejecucion anterior.
# Comentar estas lineas si se quiere re-calcular los vocabularios o si los vocabularios todavia no se han creado
#CB_SIFT=cPickle.load(open(codebook_filename_SIFT,'r'))
#CB_COLOR=cPickle.load(open(codebook_filename_COLOR,'r'))

# Carga de las 2 descripciones BoW del conjunto de aprendizaje previamente creadas y guardadas en disco en una ejecucion anterior.
# Comentar estas lineas si se quiere re-calcular la representacion o si la representacion todavia no se ha creado
#VW_SIFT_train=cPickle.load(open(visual_words_filename_train_SIFT,'r'))
#VW_COLOR_train=cPickle.load(open(visual_words_filename_train_COLOR,'r'))

# Obtiene las 2 descripciones BoW (SIFT, color) de las imagenes del conjunto de test
VW_SIFT_train=getAndSaveBoVWRepresentation(DSC_PCA_SIFT_train,k,CB_SIFT,visual_words_filename_train_SIFT)
VW_COLOR_train=getAndSaveBoVWRepresentation(CDSC_PCA_COLOR_train,k,CB_COLOR,visual_words_filename_train_COLOR)

VW_SIFT_test=getAndSaveBoVWRepresentation(DSC_PCA_SIFT_test,k,CB_SIFT,visual_words_filename_test_SIFT)
VW_COLOR_test=getAndSaveBoVWRepresentation(CDSC_PCA_COLOR_test,k,CB_COLOR,visual_words_filename_test_COLOR)

# Entrena un clasificador SVM con las imagenes del conjunto de aprendizaje y lo evalua utilizando las imagenes del conjunto de test
# para las 2 descripciones (SIFT, color)
# Devuelve la accuracy como medida del rendimiento del clasificador
ac_BOVW_SIFT = trainAndTestLinearSVM(VW_SIFT_train,VW_SIFT_test,GT_ids_train,GT_ids_test,C)
ac_BOVW_COLOR = trainAndTestLinearSVM(VW_COLOR_train,VW_COLOR_test,GT_ids_train,GT_ids_test,C)

# Entrena un clasificador SVM con la descripcion SIFT.
# Al incluir el parametro "probability=True", podremos luego recuperar la probabilidad asociada al resultado de la clasificacion para poder aplicar un esquema de late fusion.
# De forma previa al aprendizaje del clasificador, los datos se re-escalan para normalizarlos a media 0 y desviacion estandar 1.
stdSlr = StandardScaler().fit(VW_SIFT_train)
VW_SIFT_train_scaled = stdSlr.transform(VW_SIFT_train)
VW_SIFT_test_scaled = stdSlr.transform(VW_SIFT_test)
clf_SIFT = svm.SVC(kernel='linear', C=1,probability=True).fit(VW_SIFT_train_scaled, GT_ids_train)
ac_BOVW_SIFT = clf_SIFT.score(VW_SIFT_test_scaled, GT_ids_test)

# Entrena un clasificador SVM con la descripcion de color.
# Al incluir el parametro "probability=True", podremos luego recuperar la probabilidad asociada al resultado de la clasificacion para poder aplicar un esquema de late fusion.
# De forma previa al aprendizaje del clasificador, los datos se re-escalan para normalizarlos a media 0 y desviacion estandar 1.
stdSlr = StandardScaler().fit(VW_COLOR_train)
VW_COLOR_train_scaled = stdSlr.transform(VW_COLOR_train)
VW_COLOR_test_scaled = stdSlr.transform(VW_COLOR_test)
clf_COLOR = svm.SVC(kernel='linear', C=1,probability=True).fit(VW_COLOR_train_scaled, GT_ids_train)
ac_BOVW_COLOR = clf_COLOR.score(VW_COLOR_test_scaled, GT_ids_test)

# Evalua los dos clasificadores entrenados previamente (para SIFT y color) con la funcion "predict_proba".
# La funcion "predict_proba" devuelve una probabilidad de clasificacion para cada una de las clases
# Para la representacion late fusion se concatenan los vectores de probabilidad con la confianza de cada una de las clases para ambos descriptores


prob_train = clf_SIFT.predict_proba(VW_SIFT_train_scaled)
prob_Color_train = clf_COLOR.predict_proba(VW_COLOR_train_scaled)
prob_test = clf_SIFT.predict_proba(VW_SIFT_test_scaled)
prob_Color_test = clf_COLOR.predict_proba(VW_COLOR_test_scaled)


prob_train_max        = []
prob_Color_train_max  = []
prob_test_max         = []
prob_Color_test_max   = []

for i in range(len(prob_train)):
    prob_train_max.append(prob_train[i].max())
    prob_Color_train_max.append(prob_Color_train[i].min())
    prob_test_max.append(prob_test[i].max())
    prob_Color_test_max.append(prob_Color_test[i].min())

prob_train_max        =  np.amax(prob_train_max)
prob_Color_train_max  =  np.amax(prob_Color_train_max)
prob_test_max         =  np.amax(prob_test_max)
prob_Color_test_max   =  np.amax(prob_Color_test_max)


late_train = np.hstack((prob_train,prob_Color_train))
late_test =  np.hstack((prob_test,prob_Color_test))

# Entrena y evalua el clasificador final a partir de la representacion late fusion
stdSlr = StandardScaler().fit(late_train)
late_train_scaled = stdSlr.transform(late_train)
late_test_scaled =  stdSlr.transform(late_test)
clf_LATE = svm.SVC(kernel='linear', C=1).fit(late_train_scaled, GT_ids_train)
ac_BOVW_LF = clf_LATE.score(late_test_scaled,GT_ids_test)


print 'Accuracy BOVW with LinearSVM SIFT: '+str(ac_BOVW_SIFT)
print 'Accuracy BOVW with LinearSVM Color: '+str(ac_BOVW_COLOR)
print 'Accuracy BOVW with LinearSVM Late Fusion: '+str(ac_BOVW_LF)

# Create Configuration

path_train = '../Databases/MIT_split_min/train/'
path_test = '../Databases/MIT_split_min/test/'

detector='Dense'
descriptor='SIFT'

descriptorColor='PROB'  # PROB. Probabilities color. HIST. Histogram

kernel = 'his'

num_samples = 50000
n_classifiers = 28

color_constancy     = 1         # If is 1 is applied
isRandomized        = 0         # If is 1, SVD randomized is applied in PCA
isCrossValidation   = 0         # If is 1, Cross-Validation is performed
folds = 10

fusion = 'intermediate'                 # 'none', 'early', 'intermediate', 'late'

spatialPyramid = 1
SPConfiguration = 3             # If is 1: I+2x2+4x4 grid. If is 2: I+3x1 grid. If is 3: I+3x3 grid. If is 4 I+2x2 grid.

K = 1000

codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
codebook_Color_filename='CB_'+detector+'_'+descriptorColor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_Color_filename_train='VW_train_'+detector+'_'+descriptorColor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_Color_filename_test='VW_test_'+detector+'_'+descriptorColor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_SPM_filename_train='VWSPM_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_SPM_filename_test='VWSPM_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_SPM_Color_filename_train='VWSPM_train_'+detector+'_'+descriptorColor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'
visual_words_SPM_Color_filename_test='VWSPM_test_'+detector+'_'+descriptorColor+'_'+str(num_samples)+'samples_'+str(K)+'centroids.dat'

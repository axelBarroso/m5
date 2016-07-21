""" 
    
    PRODUCT QUANTIZATION
    
    Master in Computer Vision - Barcelona
    
    Author: Francesco Ciompi
         
"""

from numpy import *
from scipy.cluster.vq import *    # for k-means
import pickle                    # save/load data files
import platform
#if not platform.node().lower().startswith('compute'):
#    from pylab import *       # matplotlib, for graphical plots
#from PIL import Image     # python image library    
import scipy.cluster.vq as vq

from mcv_tools import *
from mcv_lsh import *
""" 
    Based on paper:

    [1] Product quantization for nearest neighbor search
        Herve Jegou, Matthijs Douze, Cordelia Schmid

    1. given the library, we first compute the codebook by splitting the descriptor into 'm' parts
    2. the same library is then quantized by applying the codebook -> product quantization
    3. given a test codeword, [...]

"""


#
#    COMPUTE THE CODEBOOK FOR VECTOR QUANTIZATION ON A DATASET
#
def computeProductQuantization(TRAINFILES,params,showFig=False,Folder=''):

    #
    # Product Quantization work with single vectors. It is then applied
    # to images described by a unique vector (e.g., GIST) or the result 
    # of a vector description of the image (Fisher Vector, Sparse coding etc.)
    #

    #    
    # get descriptors
    #    
    X = [None]*len(TRAINFILES)
    for nfile in range(0,len(TRAINFILES)):
        print TRAINFILES[nfile]
        f = open(TRAINFILES[nfile], 'rb');
        X[nfile] = pickle.load(f)
        f.close()
    X = squeeze(np.array(X))


    #
    # compute and store codebook
    #
    CODEBOOK = compute_codebook(X,params['pq_m'],params['pq_K'],params['global'],Folder)
    

    #
    # apply product quantization to dataset and store descriptors
    #
        
    n_im = -1
    for imfile in TRAINFILES:
        n_im += 1

        qv = quantize_vector(X[n_im],params['pq_m'],CODEBOOK,params['pq_K'])
        f = open(Folder+'pq_'+imfile.split('/')[-1][0:-4]+'.pkl', 'wb');
        pickle.dump(qv,f)
        f.close()


#
#    COMPUTE THE CODEBOOK FOR VECTOR QUANTIZATION
#
def compute_codebook(X,m,k,descr,Folder):
    """ compute the codebook 'C' for input data matrix 'X' and number of splits 'm' 
    codebook is computed with k-means clustering, with parameter 'k' """
    
    D = shape(X)[1] # dimensionality of descriptor
    D_star = D/m    # size of subvectors
    if mod(D,m) != 0:
        print 'ERROR: descriptor size not an integer multiple of subvectors number!'
        exit(-1)
    
    CODEBOOK = zeros((k*m,D_star))
    
    for i in range(m):
        x = X[:,D_star*i:D_star*(i+1)]
        print 'create codebook of '+str(k)+' words for '+str(i)+'/'+str(m)+' data size = '+str(shape(x))
        
        centroids,variance = vq.kmeans2(x,k,minit='points')
        print "CODEBOOK shape: " + str(shape(CODEBOOK))
        print "CODEBOOK shape ["+str(i*k)+":"+str((i+1)*k)+"] - centroids shape ["+ str(shape(centroids))+"]"    
        #print centroids    
        CODEBOOK[i*k:(i+1)*k] = centroids
        print 'stored '+str(k)+' centroids for subvector '+str(i)
    
        
    f = open(Folder+'PQ_codebook_'+descr+'.pkl', 'wb')
    pickle.dump(CODEBOOK,f);
    f.close()
    return CODEBOOK


#
#    APPLY VECTOR QUANTIZATION TO A VECTOR 'v'
#
def quantize_vector(v,m,codebook,k):
    """ 
    vector 'v', split into 'm' parts 
    """
    v = reshape(v,(1,-1))
    D = shape(v)[1]         # dimensionality of descriptor
    D_star = D/m    # size of subvectors    
        
    CODE = zeros((m,1))    
        
    for i in range(m):
        u = v[:,D_star*i:D_star*(i+1)]    # eq.(8) in [1]
        C_i = codebook[i*k:(i+1)*k]
        code,distance = vq.vq(u,C_i)
        CODE[i] = code    # eq.(9) in [1]
    
    return CODE.astype(int)
    
        
    

#
#    COMPUTE DISTANCE BETWEEN TWO VECTORS USING VECTOR QUANTIZATION
#
def pq_distance(x,y,dist_mode,codebook,m,k,vec_mode='coded'):

    # the sqrt is avoided for practical computation (as stated in [1])
    # since sqrt is a monotonically increasing function
    
    # yq is the already quantized vector, fetched from the database
    
    # vec_mode:
    #    'coded': x is full vector and y is the coded vector
    #    'full': x and y are full vectors

    if x.size != y.size and vec_mode != 'coded':
        print 'ERROR: the two vectors must be equal in size!'
        exit(-1)

    D = x.size         # dimensionality of descriptor
    D_star = D/m    # size of subvectors

    if vec_mode != 'coded':
        code_y = quantize_vector(y,m,codebook,k)
    else:
        code_y = y
       
    if dist_mode == 'sdc':
    
        # symmetric distance computation, eq.(12) in [1]
        code_x = quantize_vector(x,m,codebook,k)
        
        d = 0.0
        for j in range(m):
            C_j = codebook[j*k:(j+1)*k]
            q_x = C_j[code_x[j]]
            q_y = C_j[code_y[j]]
            d+=sum((q_x-q_y)**2)
        
        return d
        
    elif dist_mode == 'adc':
    
        # asymmetric distance computation, eq.(13) in [1]
        d = 0.0
        for j in range(m):
            C_j = codebook[j*k:(j+1)*k]
            u_x = ravel(x)[D_star*j:D_star*(j+1)]
            q_y = ravel(C_j[code_y[j]])
            d+=sum((u_x-q_y)**2)
        
        return d
            

    
#
#    compute query product-quantization
#    
def pq_query(x,CODEBOOK,params,LIBRARY,lshDir):

    # compute query results
    if params['lsh']:
        # load pre-compute buckets
        f = open(lshDir+'lsh_G.pkl', 'rb');
        G = pickle.load(f)
        C = pickle.load(f)
        f.close()
                                
    vdist = zeros((1,len(LIBRARY)))
            
    # scan the library
    cont = 0
    qq = quantize_vector(x,params['pq_m'],CODEBOOK,params['pq_K'])
    for pqfile in LIBRARY:
                
        f = open(pqfile,'rb')
        qv = pickle.load(f)
        f.close()                
                
        if params['lsh']:
            # compute distance with locality sensitive hashing

            vdist[0,cont] = lsh_distance(qq,qv,params,G,C)
        else:                
                
            # compute distance with product quantization
            vdist[0,cont] = pq_distance(x,qv,params['pq_dist_mode'],CODEBOOK,params['pq_m'],params['pq_K'])
            
        cont+=1
            
    # retrieve the most similar
    idxs = argsort(vdist)[0]
    
    return idxs





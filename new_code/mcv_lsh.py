"""
	Locality Sensitive Hashing (LSH)

	Master in Computer Vision - Barcelona
	
	Author: Francesco Ciompi, Ramon Baldrich
		
"""

import pickle					# save/load data files
import mcv_tools as tl
from numpy import *
import random as rnd
#import platform
#if not platform.node().lower().startswith('compute'):
#    from pylab import *       # matplotlib, for graphical plots

"""

	Based on the paper:
	[1] A. Gionis, P. Indyk, R. Motwani, "Similarity Search in High Dimensions via Hashing"
	

"""


#
#	compute lsh descriptors from product quantization codes
#
def computeLSH(filesDir,C,params,lshDir):
    # C is the maximum value of all the coordinates of "p"	


    files = tl.getFilelist(filesDir,'pkl')[0]			# "mcvtools."
    f = open(files[1], 'rb')
    feat = pickle.load(f)
    f.close()
    
	
    l = params['lsh_l']			# number of buckets/hashing tables	
    nmax = params['lsh_maxbits']	     # maximum number of bits per bucket
    d = feat.shape[1]          # data dimensionality
	
	# generate (and store) buckets
    G = buckets(C,l,d,nmax)
    print 'Random buckets: '
    print str(G)+'\n'
	
    f = open(lshDir+'LSH_G.pkl', 'wb');
    pickle.dump(G,f)
    pickle.dump(C,f)
    f.close()
	
	#
	# pre-processing of descriptor 
	#
	# this step is Algorithm in Figure (1) in [1]
	#
    for filename in files:
	
        print 'processing file '+filename

        if  (len(filename.split('codebook'))>1):
            continue
		
        f = open(filename, 'rb')
        feat = pickle.load(f)
        f.close()
        # preprocessing of the vector (section 3.1 in [1])
        g = preproc(feat,C,G)
		
				
        f = open(lshDir+'ls_'+filename.split('/')[-1][0:-4]+'.pkl', 'wb');
        pickle.dump(g,f)
#		pickle.dump(G,f)
        f.close()	

	print "Computing LSH finished!"
	


#
#	generate buckets / hasing tables
#
def buckets(C,l,d,nmax):
	
	G = []
	
	for j in range(l):
		
		G.append(rnd.sample(xrange(C*d), rnd.randint(1,nmax)))		
		
	return G
	
	
#
#	from integer to unary code
#
#	this is the functional "v" in [1]	
def unaryC(x,C):

	y = zeros((1,C))
	y[0,0:x] = 1
	
	return y

#
#	lsh preprocess descriptor
#	
def preproc(p,C,G):
	
	# joint unary version 'v(p)' of coordinates in 'p'
	vp = []
	for cw in p:
		vp.append(unaryC(cw+1,C))		
	vp = array(vp).flatten()
		
	# projection on buckets
	g = []
	for I in G:
		g.append(vp[array(I)-1]) 
	
	return g


    
#
#	Compute LSH distance between a new descriptor 'q' and a library lsh descriptor 'p'
#	The descriptor 'q' is first quantized and then preprocessed
#
def lsh_distance(q,vp,params,G,C):
    
	# the quantized version of 'q' is pre-processed
	vq = preproc(q,C,G)
	
	# the Hamming distance between p and q through LSH is computed
	d = 0
	for j in range(len(vp)):
		d+= tl.hamming(vp[j],vq[j])
		
	return d		


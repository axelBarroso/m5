""" 
	
     Image descriptors
	
     Master in Computer Vision - Barcelona
 
    Author: Francesco Ciompi, Ramon Baldrich, Jordi Gonzalez
		 
"""

from numpy import *	# numpy, for maths computations
from scipy.ndimage import filters
#from pylab import *	# matplotlib, for graphical plots
#import opencv
#import cv
from skimage import feature
# import leargist

# Histogram of Gradient (from 'skimage')
def hog(img):

	S = shape(img)
	HOG = feature.hog(double(img),orientations=9,pixels_per_cell=(S[0]/3,S[1]/3),cells_per_block=(3,3),visualise=True)
	return HOG[0]

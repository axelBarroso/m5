import cv2
import numpy as np
import time

ima=cv2.imread('../Databases/MIT_split/train/forest/cdmc101.jpg')
gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)

detector=cv2.FeatureDetector_create('SIFT')
descriptor = cv2.DescriptorExtractor_create('SIFT')
	
print 'Extracting Keypoints'
init=time.time()
kpts=detector.detect(gray)
end=time.time()
print 'Extracted '+str(len(kpts))+' keypoints.'
print 'Done in '+str(end-init)+' secs.'
print ''
print 'Computing SIFT descriptors'
init=time.time()
kpts,des=descriptor.compute(gray,kpts)
end=time.time()
print 'Done in '+str(end-init)+' secs.'
ggg = []
ggg.append(kpts[0])
im_with_keypoints = cv2.drawKeypoints(ima, ggg, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey()



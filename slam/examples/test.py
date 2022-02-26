import cv2 as cv
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('/home/daniel/Desktop/kitty-slam/data/KITTI_sequence_1/image_l/000000.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('/home/daniel/Desktop/kitty-slam/data/KITTI_sequence_1/image_r/000001.png',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.ORB_create(2000)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 2) #2
search_params = dict(checks=100)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

print(des1.shape)
# des1 = np.array(des1)
# des2 = np.array(des2)
print(des1.dtype)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
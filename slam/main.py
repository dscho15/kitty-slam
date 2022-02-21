from json import load
from operator import index
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2 as cv


class VisualOdometry:

    def __init__(self, data_path, n_features):
        self.poses = self._load_poses(os.path.join(data_path, "poses.txt"))
        self.l_cam = self._load_cam(os.path.join(data_path, "calib.txt"))
        self.l_imgs = self._load_imgs(os.path.join(data_path, "image_l"))
        self.orb = cv.ORB_create(n_features)
    
    @staticmethod
    def _load_poses(file_path: str):
        
        poses = np.loadtxt(file_path, delimiter=" ").reshape((-1, 3, 4))
        return poses

    @staticmethod
    def _load_cam(file_path: str):

        cam_param = np.loadtxt(file_path, delimiter=" ")
        l_cam = cam_param[0].reshape((3, 4))
        r_cam = cam_param[1].reshape((3, 4))
        return l_cam, r_cam

    @staticmethod
    def _load_imgs(cam_dir):
        
        list_dir = np.sort(os.listdir(cam_dir))
        l_imgs = []
        for i in range(len(list_dir)):
            l_imgs.append(cv.imread(cam_dir + "/" + list_dir[i], 0))
        return l_imgs

    def compute_orb_features(self, img_1, img_2, n_features=200, vis=False):

        orb = cv.ORB_create(n_features)
        kp_1, des_1 = orb.detectAndCompute(img_1, None)
        kp_2, des_2 = orb.detectAndCompute(img_2, None)
        return (kp_1, np.float32(des_1)), (kp_2, np.float32(des_2))

    def match_orb_features(self, img_1, kp_1, des_1, img_2, kp_2, des_2):

        FLANN_INDEX_TREE = 1
        index_params = dict(algorithm=FLANN_INDEX_TREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_1, des_2, k=2)
        
        # https://www.programcreek.com/python/example/89342/cv2.drawMatchesKnn
        good_matches = []
        for m, n in matches: 
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
                
        print("Number of matches found: %d" % (len(good_matches)))
        imgs3 = cv.drawMatchesKnn(img_1, kp_1, img_2, kp_2, good_matches, None)
        
        plt.imshow(imgs3)
        plt.show()

p_data = os.path.dirname(__file__) + "/../data/KITTI_sequence_1"
visual_odometry = VisualOdometry(p_data, 2000)

img_1 = visual_odometry.l_imgs[0]
img_2 = visual_odometry.l_imgs[1]

(kp_1, des_1), (kp_2, des_2) = visual_odometry.compute_orb_features(img_1, img_2)
visual_odometry.match_orb_features(img_1, kp_1, des_1, img_2, kp_2, des_2)


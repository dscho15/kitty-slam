from json import load
from operator import index
import numpy as np
import os
import logging
from matplotlib import pyplot as plt
import cv2 as cv

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class VisualOdometry:

    FLANN_INDEX_TREE = 1
    TSH_ORB_MATCHING = 0.75

    def __init__(self, data_path, n_features=2000, n_trees=5, flann_precision=50):
        # log to a file
        logging.info("Loaded file")

        # poses, l_cam and l_imgs
        self.poses = self._load_poses(os.path.join(data_path, "poses.txt"))
        self.l_cam = self._load_cam(os.path.join(data_path, "calib.txt"))
        self.l_imgs = self._load_imgs(os.path.join(data_path, "image_l"))

        # init orb, could include more params
        self.orb = cv.ORB_create(n_features)

        # utilizes a kd-tree | trees, leaf_max_size
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        index_params = dict(algorithm=self.FLANN_INDEX_TREE, trees=n_trees)
        search_params = dict(checks=flann_precision)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def _load_poses(file_path: str):
        logging.info("Loaded file %s" % (file_path))
        poses = np.loadtxt(file_path, delimiter=" ")
        dummy_row = np.ones((len(poses), 4)) * np.array([0, 0, 0, 1])
        poses = np.hstack((poses, dummy_row)).reshape((-1, 4, 4))
        return poses

    @staticmethod
    def _load_cam(file_path: str):
        logging.info("Loaded file %s" % (file_path))
        cam_param = np.loadtxt(file_path, delimiter=" ")
        l_cam = cam_param[0].reshape((3, 4))
        r_cam = cam_param[1].reshape((3, 4))
        return l_cam, r_cam

    @staticmethod
    def _load_imgs(cam_dir: list):
        list_dir = np.sort(os.listdir(cam_dir))
        l_imgs = []
        for i in range(len(list_dir)):
            l_imgs.append(cv.imread(cam_dir + "/" + list_dir[i], 0))
        logging.info("Number of imgs found: %d" % (len(l_imgs)))
        return l_imgs

    def match_orb_features(self, i: int, visualize=False):
        img_1 = self.l_imgs[i]
        img_2 = self.l_imgs[i + 1]

        kp_1, des_1 = self.orb.detectAndCompute(img_1, None)
        kp_2, des_2 = self.orb.detectAndCompute(img_2, None)

        matches = self.flann.knnMatch(
            np.float32(des_1), np.float32(des_2), k=2)

        kps = dict(kp_1=[], kp_2=[])
        if visualize is True:
            mask_good_matches = np.zeros_like(matches)
        for i in range(len(matches)):
            # a smaller distance means closer, resource: https://www.programcreek.com/python/example/89342/cv2.drawMatchesKnn
            if matches[i][0].distance < self.TSH_ORB_MATCHING * matches[i][1].distance:
                if visualize is True:
                    mask_good_matches[i, 0] = 1
                kps["kp_1"].append(kp_1[matches[i][0].queryIdx].pt)
                kps["kp_2"].append(kp_2[matches[i][0].trainIdx].pt)

        if visualize is True:
            logging.info("Number of good matches: %d" %
                         (np.sum(mask_good_matches)))
            imgs3 = cv.drawMatchesKnn(
                img_1, kp_1, img_2, kp_2, matches[mask_good_matches], None)
            plt.imshow(imgs3)
            plt.show()

        return kps

    def estimate_essential_matrix(self):
        logging.info("Computing the essential matrix")
        pass


p_data = os.path.dirname(__file__) + "/../data/KITTI_sequence_1"
vo = VisualOdometry(p_data)
vo.match_orb_features(0)

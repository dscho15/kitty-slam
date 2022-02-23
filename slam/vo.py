from json import load
from nis import match
from operator import index
import numpy as np
import os
import logging
from matplotlib import pyplot as plt
import cv2 as cv

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class VisualOdometry:

    FLANN_INDEX_TREE = 1
    TSH_ORB_MATCHING = 0.30

    def __init__(self, data_path, n_features=2000, n_trees=5, flann_precision=50, debug=False):
        # log to a file
        logging.info("Loading poses, camera matrices and images")
        
        self.poses = self._load_poses(os.path.join(data_path, "poses.txt"))
        self.left_cam, self.right_cam = self._load_cam(os.path.join(data_path, "calib.txt"))
        self.left_imgs = self._load_imgs(os.path.join(data_path, "image_l"))
        self.right_imgs = self._load_imgs(os.path.join(data_path, "image_r"))
        self.debug = debug

        # init orb, could include more params
        self.n_features = n_features
        self.orb = cv.ORB_create(self.n_features)
        self.features = dict()
        self.matches = dict()

        # utilizes a kd-tree | trees, leaf_max_size, more information found here: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        index_params = dict(algorithm=self.FLANN_INDEX_TREE, trees=n_trees)
        search_params = dict(checks=flann_precision)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def _load_poses(file_path: str):
        logging.info("Loaded poses: %s" % (file_path))
        poses = np.loadtxt(file_path, delimiter=" ")
        dummy_row = np.ones((len(poses), 4)) * np.array([0, 0, 0, 1])
        poses = np.hstack((poses, dummy_row)).reshape((-1, 4, 4))
        return poses

    @staticmethod
    def _load_cam(file_path: str):
        logging.info("Loaded cams: %s" % (file_path))
        cam_param = np.loadtxt(file_path, delimiter=" ")
        l_cam = cam_param[0].reshape((3, 4))
        r_cam = cam_param[1].reshape((3, 4))
        return l_cam, r_cam

    @staticmethod
    def _load_imgs(cam_dir: str):
        logging.info("Load imgs: %s " % (cam_dir))
        list_dir = np.sort(os.listdir(cam_dir))
        imgs = []
        for i in range(len(list_dir)):
            imgs.append(cv.imread(cam_dir + "/" + list_dir[i], 0))
        return imgs
    
    @staticmethod
    def _H(R: np.array, t: np.array):
        H = np.eye(4, dtype=np.float32)
        H[:3, :3] = R
        H[:3, 3] = t.reshape(-1)
        return H
    
    def key(self, i: int, j: int):
        if i > j:
            return str(j) + "_" + str(i)
        return str(i) + "_" + str(j)
    
    def _find_orb_features(self, i: int):
        kp, des = self.orb.detectAndCompute(self.left_imgs[i], None)
        if self.debug:
            img = cv.drawKeypoints(self.left_imgs[i], kp, None); plt.imshow(img); plt.show()
        _kp = np.zeros((len(kp), 2))
        for idx, ele in enumerate(kp):
            _kp[idx] = ele.pt
        return {"idx": i, "keypoint": _kp, "descriptor": np.float32(des)}
    
    def _match_two_features(self, i: int, j: int):
        des_i = self.features[str(i)]["descriptor"]
        des_j = self.features[str(j)]["descriptor"]
        matches = self.flann.knnMatch(des_i, des_j, k=2)
        filtered_matches = []
        for k in range(len(matches)):
            if matches[k][0].distance < self.TSH_ORB_MATCHING * matches[i][1].distance: # smaller means better
                filtered_matches.append(np.r_[matches[k][0].queryIdx, matches[k][0].trainIdx])
        return {"i": i, "j": j, "matches": np.stack(filtered_matches)} 
    
    def _find_orb_features_and_match(self, i: int, j: int):
        keys = [i, j]
        for _, key in enumerate(keys):
            if str(key) not in self.features:
                self.features[str(key)] = self._find_orb_features(key)  
        key = self.key(i, j)
        if key not in self.matches:
            self.matches[key] = self._match_two_features(i, j)
        logging.info("Number of matches, between image %s and %s: %d" % (str(i), str(j), len(self.matches[key]["matches"])))
    
    def estimate_essential_matrix(self, i: int, j: int):
        self._find_orb_features_and_match(i, j)
        kp_i, kp_j = self.features[str(i)]["keypoint"], self.features[str(j)]["keypoint"]
        _, _, matches = self.matches[self.key(i, j)].values()
        q_i, q_j = kp_i[matches[:, 0]], kp_j[matches[:, 1]]
        E, mask = cv.findEssentialMat(q_i, q_j, self.left_cam[:3, :3])
        return q_i, q_j, E
        
    def decompose_essential_matrix(self, q_i, q_j, E):
        R1, R2, t = cv.decomposeEssentialMat(E)
        KA = self.left_cam
        homogenous_transformations = [self._H(R1, t), self._H(R1, -t), self._H(R2, t), self._H(R2, -t)]
        best_pair = np.zeros(len(homogenous_transformations))
        for idx, H in enumerate(homogenous_transformations):
            Pi = KA @ np.eye(4, 4)
            Pj = KA @ H
            Q = cv.triangulatePoints(Pi, Pj, q_i.T, q_j.T)
            Q = Q / Q[3, :]
            z = np.array([0, 0, 1]).reshape((1, 3))
            z_validation_cam1 = (z @ np.eye(3, 4) @ Q) > 0
            z_validation_cam2 = (z @ H[:3, :] @ Q) > 0
            z_validation = np.sum(np.logical_and(z_validation_cam1, z_validation_cam2))
            best_pair[idx] = z_validation
        idx = np.argmax(best_pair)
        return homogenous_transformations[idx][:3, :3], homogenous_transformations[idx][:3, 3]

p_data = os.path.dirname(__file__) + "/../data/KITTI_sequence_1"
vo = VisualOdometry(p_data, n_features=2000, debug=False)
q_i, q_j, E = vo.estimate_essential_matrix(0, 1)
R, t = vo.decompose_essential_matrix(q_i, q_j, E)

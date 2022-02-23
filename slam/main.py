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
    TSH_ORB_MATCHING = 0.5

    def __init__(self, data_path, n_features=2000, n_trees=5, flann_precision=50, debug=True):
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
    def _load_imgs(cam_dir: list):
        logging.info("Load imgs: %s " % (cam_dir))
        list_dir = np.sort(os.listdir(cam_dir))
        left_imgs = []
        for i in range(len(list_dir)):
            left_imgs.append(cv.imread(cam_dir + "/" + list_dir[i], 0))
        logging.info("Number of imgs registered: %d" % (len(left_imgs)))
        return left_imgs

    @staticmethod
    def _H(R: np.array, t: np.array):
        H = np.eye(4, dtype=np.float32)
        H[:3, :3] = R
        t[:3, 3] = t
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
        mask = np.zeros(len(matches), dtype=np.int32)
        for k in range(len(matches)):
            if matches[k][0].distance < self.TSH_ORB_MATCHING * matches[i][1].distance: # smaller means better
                mask[k] = 1
        return {"i": i, "j": j, "matches": matches, "mask": mask} 
    
    def _find_orb_features_and_match(self, i: int, j: int):
        keys = [i, j]
        for _, key in enumerate(keys):
            if str(key) not in self.features:
                self.features[str(key)] = self._find_orb_features(key)  
        key = self.key(i, j)
        if key not in self.matches:
            self.matches[key] = self._match_two_features(i, j)
        logging.info("Number of matches, between image %s and %s: %d" % (str(i), str(j), np.sum(self.matches[str(i) + "_" + str(j)]["mask"])))
    
    def _estimate_essential_matrix(self, i: int, j: int):
        self._find_orb_features_and_match(i, j)
        kp_i, kp_j = self.features[str(i)]["keypoint"], self.features[str(j)]["keypoint"]
        key = self.key(i, j)
        _, _, matches, mask = self.matches[key].values()
        kp_i, kp_j = kp_i[mask], kp_j[mask]
        
        
        
        
        # cv.findEssentialMat()

# def match_orb_features(self, i: int, visualize=False):
        
    #     img_1 = self.l_imgs[i]
    #     img_2 = self.l_imgs[i + 1]

    #     kp_1, des_1 = self.orb.detectAndCompute(img_1, None)
    #     kp_2, des_2 = self.orb.detectAndCompute(img_2, None)

    #     matches = self.flann.knnMatch(np.float32(des_1), np.float32(des_2), k=2)

    #     kps = dict(kp_1=[], kp_2=[])
    #     if visualize is True:
    #         mask_good_matches = np.zeros_like(matches)
    #     for i in range(len(matches)):
    #         # a smaller distance means closer, resource: https://www.programcreek.com/python/example/89342/cv2.drawMatchesKnn
    #         if matches[i][0].distance < self.TSH_ORB_MATCHING * matches[i][1].distance:
    #             if visualize is True:
    #                 mask_good_matches[i, 0] = 1
    #             kps["kp_1"].append(kp_1[matches[i][0].queryIdx].pt)
    #             kps["kp_2"].append(kp_2[matches[i][0].trainIdx].pt)

    #     if visualize is True:
    #         logging.info("Number of good matches: %d" %
    #                      (np.sum(mask_good_matches)))
    #         imgs3 = cv.drawMatchesKnn(
    #             img_1, kp_1, img_2, kp_2, matches[mask_good_matches], None)
    #         plt.imshow(imgs3)
    #         plt.show()

    #     return kps

    # def estimate_essential_matrix(self):
    #     logging.info("Computing the essential matrix")
    #     pass


p_data = os.path.dirname(__file__) + "/../data/KITTI_sequence_1"
vo = VisualOdometry(p_data, n_features=2000, debug=True)
vo._estimate_essential_matrix(0, 1)

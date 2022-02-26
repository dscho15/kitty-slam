import numpy as np
import os
from display import Display
import logging
from matplotlib import pyplot as plt
import cv2 as cv


class VisualOdometry:

    FLANN_INDEX_LSH = 6
    TSH_ORB_MATCHING = 0.5
    COLORS = False

    def __init__(self, data_path, n_features=3000, flann_precision=100, debug=False):
        
        self.gt_poses = self._load_poses(os.path.join(data_path, "poses.txt"))
        self.left_cam, self.right_cam = self._load_cam(os.path.join(data_path, "calib.txt"))
        self.left_imgs = self._load_imgs(os.path.join(data_path, "image_l"))
        self.right_imgs = self._load_imgs(os.path.join(data_path, "image_r"))
        self.debug = debug
        
        # cur pose
        self.pose = np.eye(4, 4)
        self.traj = [self.pose]
        self.it = 1

        # triangulated_points
        self.Q_3d = []
        if self.COLORS is True:
            self.colors = []

        # init orb, could include more params
        self.n_features = n_features
        self.orb = cv.ORB_create(self.n_features)
        self.features = dict()
        self.matches = dict()

        # utilizes a kd-tree | trees, leaf_max_size, more information found here: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        index_params= dict(algorithm = self.FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 2)
        search_params = dict(checks = flann_precision)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def _load_poses(file_path: str):
        poses = np.loadtxt(file_path, delimiter=" ")
        dummy_row = np.ones((len(poses), 4)) * np.array([0, 0, 0, 1])
        poses = np.hstack((poses, dummy_row)).reshape((-1, 4, 4))
        return poses

    @staticmethod
    def _load_cam(file_path: str):
        cam_param = np.loadtxt(file_path, delimiter=" ")
        l_cam = cam_param[0].reshape((3, 4))
        r_cam = cam_param[1].reshape((3, 4))
        return l_cam, r_cam

    @staticmethod
    def _load_imgs(cam_dir: str):
        list_dir = np.sort(os.listdir(cam_dir))
        imgs = []
        for i in range(len(list_dir)):
            imgs.append(cv.imread(cam_dir + "/" + list_dir[i], 0))
        return imgs
    
    @staticmethod
    def __H(R: np.array, t: np.array):
        H = np.eye(4, dtype=np.float32)
        H[:3, :3] = R
        H[:3, 3] = t.reshape(-1)
        return H
    
    def key(self, i: int, j: int):
        if i > j:
            return str(j) + "_" + str(i)
        else:
            return str(i) + "_" + str(j)
    
    def _find_orb_features(self, i: int):
        kp, des = self.orb.detectAndCompute(self.left_imgs[i], None)
        if self.debug:
            img = cv.drawKeypoints(self.left_imgs[i], kp, None); 
            plt.imshow(img); 
            plt.show()
        kp_ = np.zeros((len(kp), 2))
        for idx, ele in enumerate(kp):
            kp_[idx] = ele.pt
        return {"idx": i, "keypoint": kp_, "descriptor": des}
    
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
    
    def estimate_essential_matrix(self, i: int, j: int):
        self._find_orb_features_and_match(i, j)
        kp_i, kp_j = self.features[str(i)]["keypoint"], self.features[str(j)]["keypoint"]
        _, _, matches = self.matches[self.key(i, j)].values()
        q_i, q_j = kp_i[matches[:, 0]], kp_j[matches[:, 1]]
        E, mask = cv.findEssentialMat(q_i, q_j, self.left_cam[:3, :3], prob=0.9999, threshold=1, method=cv.RANSAC)
        mask = (mask == 1).reshape(-1)
        return q_i[mask, :], q_j[mask, :], E
        
    def decompose_essential_matrix(self, q_i, q_j, E):
        R1, R2, t = cv.decomposeEssentialMat(E)
        KA = self.left_cam
        homogenous_transformations = [self.__H(R1, t), self.__H(R1, -t), self.__H(R2, t), self.__H(R2, -t)]
        feasible_poses = np.zeros(len(homogenous_transformations))
        Q_ = []
        for idx, H in enumerate(homogenous_transformations):
            Pi = KA @ np.eye(4, 4)
            Pj = KA @ H
            Q = cv.triangulatePoints(Pi, Pj, q_i.T, q_j.T)
            Q = Q / Q[3, :]
            Q_.append(Q)
            z = np.array([0, 0, 1]).reshape((1, 3))
            z_validation_cam1 = (z @ np.eye(3, 4) @ Q) > 0
            z_validation_cam2 = (z @ H[:3, :] @ Q) > 0
            z_validation = np.sum(np.logical_and(z_validation_cam1, z_validation_cam2))
            feasible_poses[idx] = z_validation
        idx = np.argmax(feasible_poses)
        return homogenous_transformations[idx][:3, :3], homogenous_transformations[idx][:3, 3], Q_[idx]
    
    def __colors(self, i: int, q_i: np.array):
        indices = np.floor(q_i).astype(np.uint32)
        colors = (self.left_imgs[i])[indices[:, 1], indices[:, 0]]
        return colors
    
    def step(self):
        assert self.it >= 1
        if self.it >= len(self.gt_poses):
            return False
        q_i, q_j, E = self.estimate_essential_matrix(self.it-1, self.it)
        R, t, Q = self.decompose_essential_matrix(q_i, q_j, E)
        if self.COLORS is True:
            self.colors.append(self.__colors(self.it, q_i))
        w_T_cam = np.linalg.inv(self.__H(R, t))
        self.Q_3d.append(((self.pose @ Q)[:3, ...].T))
        self.pose = self.pose @ w_T_cam
        self.it += 1
        return True

if __name__ == "__main__": 

    import time
    
    disp = Display()

    p_data = os.path.dirname(os.path.abspath(__file__)) + "/../data/KITTI_sequence_2"
    vo = VisualOdometry(p_data, n_features=3000)

    fps = 10
    dt = 1/fps
    gt_poses, est_pose = [], []
    i = 0

    while vo.step() is True:
        pts = {"est_pose": None, "gt_poses": None, "cam_pts": None}
        est_pose += [np.r_[vo.pose[:3, 3]]]
        gt_poses += [vo.gt_poses[i][:3, 3]]
        pts["est_pose"] = est_pose
        pts["gt_poses"] = gt_poses
        pts["cam_pts"] = np.vstack(vo.Q_3d).reshape((-1, 3)).tolist()
        # pts["colors"] = np.hstack(vo.colors).reshape((-1)).tolist()
        disp.q.put(pts)
        time.sleep(dt)
        i += 1

    while True:
        disp.q.put(pts)
        time.sleep(dt)
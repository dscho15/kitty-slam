import os
from tarfile import BLOCKSIZE
import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

class OpticalFlow:
    
    TILE_WIDTH = 50
    TILE_HEIGHT = 50
    FLOW = 10
    
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7)

    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 3,
                      flags = cv.MOTION_AFFINE,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.03))

    bm_params = dict( minDisparity = 0,
                      numDisparities = 32, 
                      blockSize = 11, 
                      P1 = 11 * 11 * 8,
                      P2 = 11 * 11 * 32)

    def __init__(self, p_data):
        
        self.gt_poses = self._load_poses(os.path.join(p_data, "poses.txt"))
        self.left_cam, self.right_cam = self._load_cam(os.path.join(p_data, "calib.txt"))
        self.left_imgs = self._load_imgs(os.path.join(p_data, "image_l"))
        self.right_imgs = self._load_imgs(os.path.join(p_data, "image_r"))
        
        self.fast_feature_detector = cv.FastFeatureDetector_create()
        self.sgbm = cv.StereoSGBM_create(**self.bm_params)
        self.p1 = None
        self.p2 = None
        self.debug = True
        
        
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
    
    def _good_features_to_track(self, frame: np.array):
        corners = cv.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        corners = np.int0(corners)
        return corners.squeeze(1)
    
    def _fast_features_to_track(self, frame: np.array, tile_height: int, tile_width: int):    
        # tsh for the detector
        self.fast_feature_detector.setThreshold(100)
        # remove edges
        kps = []
        for col in range(tile_width, frame.shape[1]-tile_width, tile_width):
            for row in range(tile_height, frame.shape[0]-tile_height, tile_height):
                tile = frame[row : (row + tile_height), col : (col + tile_width) ]
                keypoints = self.fast_feature_detector.detect(tile, None)
                for idx, keypoint in enumerate(keypoints):
                    if idx == 10:
                        break
                    keypoint.pt = (col + keypoint.pt[0], row + keypoint.pt[1])
                    if self.debug is True:
                        kps.append(keypoint)
                    else:
                        kps.remove(keypoint.pt)
        # check for debug // visualization
        if self.debug is True:
            keypoints = tuple(kps)
            #frame = cv.drawKeypoints(frame, keypoints, None) 
            #cv.imshow("frame", frame)
            #cv.waitKey()
            for idx, keypoint in enumerate(kps):
                kps[idx] = keypoint.pt    
        kps = np.int0(np.stack(kps))
        return kps
        
    def _flow(self, frame_i: np.array, frame_j: np.array):
        # detect features that should be tracked
        self.p1 = self._fast_features_to_track(frame_i, self.TILE_WIDTH, self.TILE_HEIGHT)
        # calculate flow
        self.p2, st, err = cv.calcOpticalFlowPyrLK(frame_i, frame_j, np.float32(self.p1), None, **self.lk_params)
        st = np.logical_and(st, err < self.FLOW).squeeze(1)
        # rule out errors
        if self.p1 is not None:
            self.p1 = self.p1[st == 1, ...]
            self.p2 = self.p2[st == 1, ...]
        # check for debug // visualization
        if self.debug is True:
            # draw mask
            mask = np.zeros_like(frame_j)
            for i, (new, old) in enumerate(zip(self.p1, self.p2)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (128, 0, 0), 2)
                frame_j = cv.circle(frame_j, (int(c), int(d)), 5, color=(255, 0, 0))
            frame_j = cv.add(mask, frame_j)
            # show img
            cv.imshow("optical_flow", frame_j)
            cv.waitKey()
    
    def _determine_right_image_pts(self, pts_i, pts_j):
        pass
    
    def step(self):
        pass
    
    

if __name__ == "__main__":
    
    p_data = os.path.dirname(os.path.abspath(__file__)) + "/../data/KITTI_sequence_2"
    
    optical_flow = OpticalFlow(p_data)
    # optical_flow._fast_features_to_track(optical_flow.left_imgs[0])
    for i in range(20):
        optical_flow._flow(optical_flow.left_imgs[i], optical_flow.left_imgs[i+1])
    
    
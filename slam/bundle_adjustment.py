import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import logging
from scipy.optimize import least_squares

class SO3:
    
    @staticmethod
    def euler_zyx_to_rotm(alpha, gamma, beta):
        return (R.from_euler('z', alpha, degrees=True) * R.from_euler('y', beta, degrees=True) * R.from_euler('x', gamma, degrees=True)).as_matrix()

    @staticmethod
    def quat_to_rotm(q: np.array):
        assert q.shape[0] == 4
        return R.from_quat(q).as_matrix()
    
    @staticmethod
    def eaa_to_rotm(x: np.array):
        assert x.shape[0] == 3
        return R.from_mrp(x).as_matrix()
    
    @staticmethod
    def rotm_to_quat(rotm: np.array):
        assert rotm.shape[0] == rotm.shape[1]
        return R.as_quat(R.from_matrix(rotm))
    
class SE3(SO3):
    
    @staticmethod
    def pose(R: np.array, t: np.array):
        T = np.eye(4, 4)
        T[:3, 3] = t
        T[:3, :3] = R
        return T
    
class PinHoleCamera:
    
    def __init__(self, fx, fy, u, v, R, t):
        self.A = np.array([fx, 0, u, 0, 0, fy, v, 0, 0, 0, 1, 0]).reshape((3, 4))
        self.H = SE3.pose(R, t)
        self.P = self.A @ self.H

class BundleAdjustment:
    
    @staticmethod
    def project_3d_to_2d(cam, X_w_h):
        p_c = cam.P @ X_w_h
        p = p_c[:2, ...] / p_c[2, :]
        return p
    
    @staticmethod
    def pose_residuals_euler_zyx(x, q_c_h, X_w_h):
        zyx_angles = x[3:]
        c_R_w = SO3.euler_zyx_to_rotm(*zyx_angles)
        c_t_w = x[:3]
        cam = PinHoleCamera(700, 700, 250, 250, c_R_w, c_t_w)
        p = BundleAdjustment.project_3d_to_2d(cam, X_w_h)
        return (q_c_h - p).reshape(-1)
    
    @staticmethod
    def pose_residuals_quat(x, q_c_h, X_w_h):
        q = x[3:] / scipy.linalg.norm(x[3:])
        c_R_w = SO3.quat_to_rotm(q)
        c_t_w = x[:3]
        cam = PinHoleCamera(700, 700, 250, 250, c_R_w, c_t_w)
        p = BundleAdjustment.project_3d_to_2d(cam, X_w_h)
        return (q_c_h - p).reshape(-1)
    


if __name__ == '__main__':
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    
    zyx_angles = [45, 20, 50]
    
    logging.info("Rotating the camera...")
    
    c_R_w = SO3.euler_zyx_to_rotm(*zyx_angles)
    c_quat_w = SO3.rotm_to_quat(c_R_w)
    c_t_w = np.r_[0, 1, 2]
    
    logging.info("Created camera translation...")
    
    # generate random 4D configurations 
    X_w = np.random.rand(3, 1000)
    X_w_h = np.vstack((X_w, np.ones((1, 1000))))
    
    logging.info("Created random configurations...")
    
    # reproject the 4D configurations into 2D coords
    cam = PinHoleCamera(700, 700, 250, 250, c_R_w, c_t_w)
    logging.info("Created pose...")
    
    # reproject the image coordinates
    q_c_h = cam.P @ X_w_h # (3, 4) @ (4, 4) @ (4, 20)
    logging.info("Reprojected the 3D points...")
    
    # homogenize the coordinates
    q_c_h = q_c_h[:2, ...] / q_c_h[2, ...] #(2, 20)
    
    # change the zyx_angles (disturabance)
    zyx_angles += np.random.rand(3,) * 100
    c_R_w = SO3.euler_zyx_to_rotm(*zyx_angles)
    c_quat_w = SO3.rotm_to_quat(c_R_w)
    c_t_w = np.r_[0, 1, 2] # (3, )
    x = np.hstack((c_t_w, c_quat_w))
    
    # make a least_squares object in scipy
    ls_lms = least_squares(fun=BundleAdjustment.pose_residuals_quat, x0=x, jac="2-point", args=(q_c_h, X_w_h), verbose=True)
    res = BundleAdjustment.pose_residuals_quat(ls_lms.x, q_c_h, X_w_h) 

    print(SE3.quat_to_rotm(ls_lms.x[3:]))
    print(SE3.quat_to_rotm(c_quat_w))
    print(np.max(res))
    
    #print(BundleAdjustment.pose_residuals_quat(ls_lms.x, q_c_h, X_w_h))


    # https://www.coursera.org/lecture/robotics-perception/bundle-adjustment-i-oDj0o
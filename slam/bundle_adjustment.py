import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from scipy.optimize import least_squares

def pose(R: np.array, t: np.array):
    T = np.eye(4, 4)
    T[:3, 3] = t
    T[:3, :3] = R
    return T

def euler_zyx(alpha, gamma, beta):
    return R.from_euler('z', alpha, degrees=True) * R.from_euler('y', beta, degrees=True) * R.from_euler('x', gamma, degrees=True)

def residuals(x, q_c_h, X_w_h):
    zyx_angles = x[:3]
    c_R_w = euler_zyx(*zyx_angles).as_matrix()
    c_t_w = x[3:]
    T = pose(c_R_w, c_t_w)
    p_c = np.eye(3, 4) @ T @ X_w_h
    p = p_c[:2, ...] / p_c[2, :]
    return (q_c_h - p).reshape(-1)


if __name__ == '__main__':
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    
    zyx_angles = [45, 20,50]
    
    logging.info("Rotating the camera...")
    
    c_R_w = euler_zyx(*zyx_angles).as_matrix()
    c_t_w = np.r_[0, 1, 2]
    
    logging.info("Created camera translation...")

    # generate random 4D configurations 
    X_w = np.random.rand(3, 20)
    X_w_h = np.vstack((X_w, np.ones((1, 20))))
    
    logging.info("Created random configurations...")

    # reproject the 4D configurations into 2D coords
    c_T_w = pose(c_R_w, c_t_w)
    
    logging.info("Created pose...")

    # reproject the image coordinates
    q_c_h = np.eye(3, 4) @ c_T_w @ X_w_h # (3, 4) @ (4, 4) @ (4, 20)
    
    logging.info("Reprojected the 3D points...")

    # homogenize the coordinates
    q_c_h = q_c_h[:2, ...] / q_c_h[2, ...] #(2, 20)
    
    # change the zyx_angles (disturabance)
    zyx_angles += np.random.rand(3,) * 1000
    c_t_w = np.r_[0, 1, 2] # (3, )
    x = np.hstack((c_t_w, zyx_angles))

    # make a least_squares object in scipy
    ls_lms = least_squares(fun=residuals, x0=x, args=(q_c_h, X_w_h), verbose=True)
    res = residuals(ls_lms.x, q_c_h, X_w_h) 
    print(ls_lms.x)
    print(np.dot(res, res))


    # https://www.coursera.org/lecture/robotics-perception/bundle-adjustment-i-oDj0o
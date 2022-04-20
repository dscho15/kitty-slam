import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from scipy.optimize import least_squares

"""
    Find the camera pose given 3D to 2D correspondences using a least-squares formulation.
"""

def pose(R: np.array, t: np.array):
    T = np.eye(4, 4)
    T[:3, 3] = t
    T[:3, :3] = R
    return T

def euler_zyx(alpha, gamma, beta):
    return R.from_euler('z', alpha, degrees=True) * R.from_euler('y', beta, degrees=True) * R.from_euler('x', gamma, degrees=True)

def residuals(x, c_q__h, w_X__h):
    c_R_w = euler_zyx(*x[:3]).as_matrix()
    c_t_w = x[3:]
    T = pose(c_R_w, c_t_w)
    p_c = np.eye(3, 4) @ T @ w_X__h
    p = p_c[:2, ...] / p_c[2, :]
    return (c_q__h - p).reshape(-1)

if __name__ == '__main__':
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    
    zyx_angles = [45, 20,50]
    c_R_w = euler_zyx(*zyx_angles).as_matrix()
    c_t_w = np.r_[0, 1, 2]
    logging.info("Created camera translation...")
    
    # generate random 4D configurations 
    w_X = np.random.rand(3, 20)
    w_X__h = np.vstack((w_X, np.ones((1, 20))))
    logging.info("Created random configurations...")
    
    # reproject the 4D configurations into 2D coords
    c_T_w = pose(c_R_w, c_t_w)
    logging.info("Created pose...")
    
    # reproject the image coordinates
    c_q__h = np.eye(3, 4) @ c_T_w @ w_X__h # (3, 4) @ (4, 4) @ (4, 20)
    logging.info("Reprojected the 3D points...")

    # homogenize the coordinates
    c_q__h = c_q__h[:2, ...] / c_q__h[2, ...] #(2, 20)
    logging.info("Homogenize the image-pts...")
    
    # change the zyx_angles (disturabance)
    zyx_angles += np.random.rand(3,) * 10 # (3, )
    c_t_w = np.r_[0, 1, 2] # (3, )
    x = np.hstack((c_t_w, zyx_angles)) # (6, )
    logging.info("Stack the pose vector")

    # make a least_squares object in scipy
    lms = least_squares(fun=residuals, x0=x, args=(c_q__h, w_X__h))
    res = residuals(lms.x, c_q__h, w_X__h)

    print("The resulting pose: ", np.array2string(lms.x, precision=2, suppress_small=True))
    print("The residual-squared: ", np.array2string(np.dot(res, res), precision=16, suppress_small=True))

    # https://www.coursera.org/lecture/robotics-perception/bundle-adjustment-i-oDj0o
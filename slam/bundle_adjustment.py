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

def residuals(x, args, kwargs):
    q_c_h = kwargs["q_c_h"]
    zyx_angles = x[:3]
    c_R_w = euler_zyx(*zyx_angles).as_matrix()
    c_t_w = x[:3]
    T = pose(c_R_w, c_t_w)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    zyx_angles = [45, 20, 30]
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
    zyx_angles += np.random.rand(3,)
    c_t_w = np.r_[0, 1, 2] # (3, )
    x = np.hstack((c_t_w, zyx_angles))

    # make a least_squares object in scipy
    ls_lms = least_squares(fun=residuals, x0=x, args=(), kwargs={"q_c_h": q_c_h})

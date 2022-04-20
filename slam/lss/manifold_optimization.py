import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

# https://github.com/HIPS/autograd

def pose(R: np.array, t: np.array):
    T = np.eye(4, 4)
    T[:3, 3] = t
    T[:3, :3] = R
    return T

def euler_zyx_to_quat(alpha, gamma, beta):
    rotm = R.from_euler('z', alpha, degrees=True) * R.from_euler('y', beta, degrees=True) * R.from_euler('x', gamma, degrees=True)
    return rotm.as_quat()

def skew(x):
    return np.array([0, -x[2], x[1], x[2], 0, -x[0], -x[1], x[0], 0]).reshape((3, 3))

zyx_angles = [35, 20, 50]

euler_zyx = euler_zyx_to_quat(*zyx_angles)

skew(zyx_angles)

R_0 = np.eye(3, 3)
w = [0.01, 0.02, 0.03]

for i in range(50):

    R_0 = R_0 @ expm(skew(w))
    print(np.linalg.det(R_0))
    print(R_0)
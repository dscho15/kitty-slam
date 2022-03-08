import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from scipy.optimize import least_squares

class SO3:
    
    @staticmethod
    def euler_zyx_to_rotm(alpha, gamma, beta):
        return R.from_euler('z', alpha, degrees=True) * R.from_euler('y', beta, degrees=True) * R.from_euler('x', gamma, degrees=True)

    @staticmethod
    def quat_to_rotm(q: np.array):
        return R.from_quat(q)
    
    @staticmethod
    def eaa_to_rotm(x: np.array):
        return R.from_mrp(x)
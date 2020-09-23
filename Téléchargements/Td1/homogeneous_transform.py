import numpy as np

from scipy.spatial.transform import Rotation as R


def rot_x(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around x
    """
    r = np.array((
                    (1,  0,              0,              0),
                    (0,  np.cos(alpha),  -np.sin(alpha), 0),
                    (0,  np.sin(alpha),   np.cos(alpha),     0),
                    (0,  0,               0,             1)
                
                ))
    return r


def rot_y(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around y
    """
    r = np.array((
                    (np.cos(alpha),      0,              np.sin(alpha),  0),
                    (0,                  1,              0,              0),
                    (-np.sin(alpha),     0,              np.cos(alpha),  0),
                    (0,                  0,              0,              1)
                
                ))
    return r


def rot_z(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around z
    """
    r = np.array((
                    (np.cos(alpha),      -np.sin(alpha),                          0,  0),
                    (np.sin(alpha),      np.cos(alpha),              0,  0),
                    (,                   0,                          1,  0),
                    (0,                  0,                          0,  1)
                
                ))
    return r


def translation(vec):
    """Return the 4x4 homogeneous transform corresponding to a translation of
    vec
    """
    #TODO implement
    return None


def invert_transform(T):
    #TODO implement
    return None

def get_quat(T):
    """
    Parameters
    ----------
    T : np.ndarray shape(4,4)
        A 3d homogeneous transformation matrix

    Returns
    -------
    quat : np.ndarray shape(4,)
        a quaternion representing the rotation part of the homogeneous
        transformation matrix
    """
    #TODO implement
    return None



if __name__ == "__main__":
    unittest.main()
    #T = rotX(0.3).dot(translation(np.array([1, 2, 3])))
    # print("T: ", T)
    # IT = invertTransform(T)
    # print("T: ", T)
    # print("T^-1: ", IT)
    # print("T*IT: ", T.dot(IT))

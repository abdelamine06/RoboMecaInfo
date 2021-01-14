import numpy as np
from scipy.spatial.transform import Rotation as R

def rot_x(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around x
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=np.double)
    return r


def rot_y(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around y
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=np.double)



def rot_z(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around z
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.double)

def translation(vec):
    """Return the 4x4 homogeneous transform corresponding to a translation of
    vec
    """ 
    return np.array([[1, 0, 0, vec[0]],
                     [0, 1, 0, vec[1]],
                     [0, 0, 1, vec[2]],
                     [0, 0, 0, 1]], dtype=np.double)


def invert_transform(T):
    I = T.copy()
    RI = T[:3, :3].transpose()
    I[:3, :3] = RI
    I[:3, 3] = -RI @ T[:3, 3]
    return I

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
    return R.from_dcm(T[:3,:3]).as_quat()
    



if __name__ == "__main__":
    unittest.main()
    T = rotX(0.3).dot(translation(np.array([1, 2, 3])))
    print("T: ", T)
    IT = invertTransform(T)
    print("T: ", T)
    print("T^-1: ", IT)
    print("T*IT: ", T.dot(IT))

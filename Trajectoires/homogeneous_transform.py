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

def dcos_sin(value, d=1, cos=True):
    #derivatives = [ [1, np.sin],[1, np.cos], [-1, np.sin], [-1, np.cos]]
    index = (d+1 if cos else d) % 4
    f = np.sin if index%2 == 0 else np.cos
    return f(value) if index < 2 else -f(value)

def d_rot_x(alpha,d=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around x
    """
    if d == 0:
        return rot_x(alpha)
    c = dcos_sin(alpha, d, True)
    s = dcos_sin(alpha, d, False)
    return np.array([[0, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 0]], dtype=np.double)
    """c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[0, 0, 0, 0],
                     [0, -s, -c, 0],
                     [0, c, -s, 0],
                     [0, 0, 0, 0]], dtype=np.double)"""

def d_rot_y(alpha,d=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around y
    """
    if d == 0:
        return rot_y(alpha)
    c = dcos_sin(alpha, d, True)
    s = dcos_sin(alpha, d, False)
    return np.array([[c, 0, s, 0],
                     [0, 0, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 0]], dtype=np.double)
    """c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[-s, 0, c, 0],
                 [0, 0, 0, 0],
                 [-c, 0, -s, 0],
                 [0, 0, 0, 0]], dtype=np.double)"""

def d_rot_z(alpha,d=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around z
    """
    if d == 0:
        return rot_z(alpha)
    c = dcos_sin(alpha, d, True)
    s = dcos_sin(alpha, d, False)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=np.double)
    """c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[-s, -c, 0, 0],
                     [c, -s, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=np.double)"""

def d_translation(vec, d=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a translation of
    vec
    """
    if d == 0:
        return translation(vec)

    v = vec / np.linalg.norm(vec)
    T = np.zeros((4,4), dtype=np.double)
    T[:3,3] = v
    return T

def d_rot_x_dt(alpha,d=1, dq=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around x
    """
    one = 1 if d == 0 else 0
    c = dcos_sin(alpha, d, True) * dq
    s = dcos_sin(alpha, d, False) * dq
    return np.array([[one, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, one]], dtype=np.double)
    """c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[0, 0, 0, 0],
                     [0, -s, -c, 0],
                     [0, c, -s, 0],
                     [0, 0, 0, 0]], dtype=np.double)"""

def d_rot_y_dt(alpha,d=1, dq=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around x
    """
    one = 1 if d == 0 else 0
    c = dcos_sin(alpha, d, True) * dq
    s = dcos_sin(alpha, d, False) * dq
    return np.array([[c, 0, s, 0],
                     [0, one, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, one]], dtype=np.double)
    """c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[-s, 0, c, 0],
                 [0, 0, 0, 0],
                 [-c, 0, -s, 0],
                 [0, 0, 0, 0]], dtype=np.double)"""

def d_rot_z_dt(alpha,d=1, dq=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a rotation of
    alpha around x
    """
    one = 1 if d == 0 else 0
    c = dcos_sin(alpha, d, True) * dq
    s = dcos_sin(alpha, d, False) * dq
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, one, 0],
                     [0, 0, 0, one]], dtype=np.double)
    """c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[-s, -c, 0, 0],
                     [c, -s, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=np.double)"""

def d_translation_dt(vec,d=1, dq=1):
    """Return the 4x4 homogeneous transform corresponding to the derivative of a translation of
    vec
    """
    if d == 0:
        return translation(vec) * dq

    v = vec / np.linalg.norm(vec) * dq
    T = np.zeros((4,4), dtype=np.double)
    T[:3,3] = v
    return T



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
    return R.from_dcm(T[:3,:3]).as_quat()



if __name__ == "__main__":
    unittest.main()

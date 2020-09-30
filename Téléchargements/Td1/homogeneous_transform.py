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
                    (np.cos(alpha),      -np.sin(alpha),                 0,  0),
                    (np.sin(alpha),      np.cos(alpha),              0,  0),
                    (0,                   0,                          1,  0),
                    (0,                  0,                          0,  1)
                
                ))
    return r


def translation(vec):
    """Return the 4x4 homogeneous transform corresponding to a translation of
    vec
    """ 
    dX = vec[0]
    dY = vec[1]
    dZ = vec[2]
    t = np.array(( 
                    (1 , 0, 0, dX),
                    (0 , 1, 0, dY),
                    (0 , 0, 1, dZ),
                    (0 , 0, 0, 1)
                ))
    return t


def invert_transform(T):
    Rot = np.zeros((3,3))
    Trans = np.zeros([3,1])
    T[3,3]=1
    T[3,2]=0
    T[3,1]=0
    T[3,0]=0
    
    for i in range(0,3):
        for j in range(0,3):
            Rot[i,j] = T[i,j] 
    for i in range(3):
        Trans[i]= T[i,3]

    Rot_t = Rot.transpose()
    Tran_t = (-Rot_t).dot(Trans)

    for i in range(0,3):
        for j in range(0,3):
            T[i,j] = Rot_t[i,j]
    for i in range(3):
            T[i,3]=Tran_t[i]

    return T

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
    T = rotX(0.3).dot(translation(np.array([1, 2, 3])))
    print("T: ", T)
    IT = invertTransform(T)
    print("T: ", T)
    print("T^-1: ", IT)
    print("T*IT: ", T.dot(IT))

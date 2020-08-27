from matplotlib.path import Path
from typing import Union

import numpy as np
import scipy
import transforms3d
import utm

def orthonormal(w:np.ndarray) -> np.ndarray:
    """Orthonormalize (square) matrix w"""
    return w.dot(scipy.linalg.inv(scipy.linalg.sqrtm(w.T.dot(w))))

def tf_eye(dtype = np.float32) -> np.ndarray:
    """4x4 Identity matrix"""
    return np.eye(4, dtype = dtype)

def tf_inv(transform:np.ndarray) -> np.ndarray:
    """
        Efficient 4x4 affine transform matrix invert
        Assumes orthonormal rotation matrix, and that tansform[3,:] == [0,0,0,1]
    """
    R = transform[:3,:3]
    T = transform[:3, 3]

    inv_tf = np.zeros_like(transform)
    inv_tf[:3,:3] = R.T 
    inv_tf[:3,3] = np.dot(-R.T, T)
    inv_tf[3,3] = 1

    return inv_tf

def RT_to_tf(R, T):
    tf = tf_eye(R.dtype)
    tf[:3, :3] = R
    tf[:3, 3] = T
    return tf

def tf_to_RT(tf):
    return tf[:3, :3], tf[:3, 3] # R, T

def batch_dot(a, b):
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    return np.einsum('ij,ij->i',a, b)

def to_homo(v):
    return np.array([v[0], v[1], v[2], 1], v.dtype)

def from_homo(v):
    return v[0:3]/v[3]

def map_point(m:np.ndarray, v:np.ndarray) ->  np.ndarray:
    '''Apply a 4x4 transform on 3x1 point
    Args:
        m: a 4x4 transform
        v: a 1x3 point
    '''
    return from_homo(np.dot(m, to_homo(v)))

def map_points(m:np.ndarray, v:np.ndarray) ->  np.ndarray:
    """Apply a 4x4 transform on 3x1 point(s)
    
    Args:
        m: a 4x4 transform
        v: Nx3 point matrix
    """
    if len(v.shape) == 1:
        return map_point(m, v)
    return (np.dot(m[:3, :3], v.T) + m[:3,3,None]).T

def map_vectors(m, v):
    return np.dot(m[:3,:3], v.T).T

def normalized(v):
    return v/np.linalg.norm(v)

def face_normals(triangles, vertices):
    """Computes a normal vector for each triangle"""
    v01 = vertices[triangles[:,1]] - vertices[triangles[:,0]]
    v12 = vertices[triangles[:,2]] - vertices[triangles[:,1]]
    n = np.cross(v01, v12) + 1e-7 #prevent true_divide error
    n /= np.linalg.norm(n, axis=-1)[:,None]
    return n

def pca(X_zero_centered):
    """Preforms a Principal Component Analysis on X_zero_centered"""
    # https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
    # Data matrix X_zero_centered, assumes 0-centered
    n, m = X_zero_centered.shape
    # assert np.allclose(X_zero_centered.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X_zero_centered.T, X_zero_centered) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)

    idx = np.argsort(eigen_vals)[::-1]

    return eigen_vals[idx], eigen_vecs[idx]

def rigid_transform_3D_SVD(A:np.ndarray, B:np.ndarray) -> np.ndarray:
    """Finds the rigid transform that maps the N points in A to the N points in B

    Args:
        A: Nx3 points
        B: Nx3 points
    Returns:
        a 4x4 (rigid) tranform matrix
    """

    A_centered = A-np.mean(A, axis=0)
    B_centered = B-np.mean(B, axis=0)

    cov = A_centered.T.dot(B_centered)/A_centered.shape[1]
    u, s, v = np.linalg.svd(cov)

    d = (np.linalg.det(u) * np.linalg.det(v)) < 0.0

    if d:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    R = u.dot(v) # as per SVD, R is orthonormal

    T = np.mean(A.T-R.dot(B.T), axis=1)

    return RT_to_tf(R, T)

def rasterize(width = 640, height = 480, polygon = [(0.1*640, 0.1*480), (0.15*640, 0.7*480), (0.8*640, 0.75*480), (0.72*640, 0.15*480)]):
    """Rasterizes (i.e. 'draws') a polygon in a (height x width) boolean matrix (i.e. a binary image)"""

    poly_path=Path(polygon)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

    mask = poly_path.contains_points(coors)

    return mask.reshape(height, width) # you can plot the result in e.g. plt.imshow()

def imu_to_utm(gps:Union[dict, np.ndarray], euler: Union[dict, np.ndarray]) -> np.ndarray:
    """IMU data (NED) to utm (ENU)

    Args:
        gps: latitude, longitude, altitude (in meters)
        euler: roll, pitch, yaw (in radians) in NED coordinates
    Returns:
        a 6x1 ndarray containing x,y,z,roll,pitch,yaw in ENU coordinates
    """
    x, y, zone_number, zone_letter = utm.from_latlon(gps['latitude'], gps['longitude']) #utm is ENU
    pose_ENU = np.array([x, y, gps['altitude']
    , euler['roll'] #around North in NED, around East in ENU
    , -euler['pitch'] #around East in NED, inverted since ENU's z coordinate points up
    , (-euler['yaw'] + np.pi / 2)]) #NED yaw is around down, ENU yaw is around up. Dephased 180deg, since NED's front is North, ENU's front is East 
    return pose_ENU 

def tf_from_poseENU(pose_ENU:Union[list, np.ndarray], dtype = np.float32) -> np.ndarray:
    """ Creates a 4x4 transform matrix from a 6x1 pose vector
    
    Args:
        pose_ENU: 6x1 array-like containing the pose (in ENU refrential)\: x,y,z,roll,pitch,yaw, where angles are in radians
    Returns:
        the 4x4 transfrom matrix
    """
    tf = tf_eye(dtype)
    tf[0:3, 3] = pose_ENU[0:3]
    roll, pitch, yaw = [pose_ENU[i+3] for i in range(3)]
    tf[0:3, 0:3] = transforms3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')
    return tf

def tf_from_pos_euler(pos:Union[list, np.ndarray] = [0,0,0], euler_deg:Union[list, np.ndarray] = [0,0,0], dtype = np.float32) -> np.ndarray:
    """ Creates a 4x4 transform matrix from a position an a triplet of euler angles
    Args:
        pos: the position (x,y,z)
        euler: the euler angles (rx, ry, rz), in degrees
    Returns:
        the 4x4 transfrom matrix
    """
    if isinstance(pos, np.ndarray):
        pos = pos.tolist()
    
    euler_deg = np.radians(euler_deg).tolist()
    
    return tf_from_poseENU(pos + euler_deg) # [a,b,c] + [d,e,f] => [a,b,c,d,e,f]

def pos_euler_from_tf(tf):
    """Convert a 4x4 transformation matrix to pose-euler (in degree)
    """
    tx,ty,tz = tf[:3,3]
    rx,ry,rz = transforms3d.euler.mat2euler(tf[:3, :3], axes='sxyz')
    return tx,ty,tz,np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)


def bbox_to_8coordinates(c_xyz,d_xyz,r_xyz):
    """ Convert kitty convention box coordinate to an array of the 8 coordinates of box coordinates in 3D
    Args:
        c_xyz: center of the box
        d_xyz: dimension x,y,z (length,width,height) of the box
        r_xyz : roll pitch yaw rotation in radians (Yaw aligned on box x dimension)
    Returns:
        array of (8,3) coordiantes
    """
    rotation = transforms3d.euler.euler2mat(r_xyz[0],r_xyz[1],r_xyz[2])
    coordinates = np.array([
                  [- d_xyz[0]/2, - d_xyz[1]/2, - d_xyz[2]/2]
                , [- d_xyz[0]/2, - d_xyz[1]/2, d_xyz[2]/2]
                , [- d_xyz[0]/2, d_xyz[1]/2, - d_xyz[2]/2]
                , [- d_xyz[0]/2, d_xyz[1]/2, d_xyz[2]/2]
                , [d_xyz[0]/2, - d_xyz[1]/2, - d_xyz[2]/2]
                , [d_xyz[0]/2, - d_xyz[1]/2, d_xyz[2]/2]
                , [d_xyz[0]/2, d_xyz[1]/2, - d_xyz[2]/2]
                , [d_xyz[0]/2, d_xyz[1]/2, d_xyz[2]/2]])
    vertices = ((np.dot(rotation, coordinates.T).T) + [c_xyz[0],c_xyz[1],c_xyz[2]])
    return vertices



def points_inside_box_mask(points, aabb, tf):
    """ Return a mask that gives True of False for each 3d point if it lies inside the bboxe

        points - (M,3) are the 3d points
        aabb - min max along x,y,z, centered at zero, shape=(2,3) [basically, this is aa=-d_xyz/2, and bb=+d_xyz/2]
        tf - transformation matrix Box_from_Datasource
    """

    Xp = map_points(tf, points)
    mask = (Xp[:,0]>=aabb[0,0]) * (Xp[:,0]<=aabb[1,0]) * (Xp[:,1]>=aabb[0,1]) * (Xp[:,1]<=aabb[1,1]) * (Xp[:,2]>=aabb[0,2]) * (Xp[:,2]<=aabb[1,2]) 
    return mask


def map_angle_to_domain(angle, domain=[0,2*np.pi]):
    """ Convert a angle (in radian) in the range given by the domain.  
    Note that the sup bound is excluded and the inf bound is included.
    Note that the length of domain must be 2pi.
        Example of use: 
            map_angle_to_a_b(angle=5*np.pi/2, domain=[0,2*np.pi])
            = np.pi/2
    """
    new_angle = np.copy(angle)
    while (new_angle<domain[0])or(new_angle>=domain[1]):
        if new_angle<domain[0]:
            new_angle += 2*np.pi
        if new_angle>=domain[1]:
            new_angle -= 2*np.pi
    return new_angle



def pcloud_inside_box(pcloud, box, margin=0):
    """Returns indices of the points from pcloud that are inside the box"""
    # Note : does a similar thing than points_insisde_box_mask(), but does not require the transformation matrix to IMU.
    P = bbox_to_8coordinates(box['c'], box['d']+[2*margin,2*margin,2*margin], box['r'])
    u, v, w = P[5]-P[1], P[3]-P[2], P[3]-P[1]
    pu, pv, pw = np.matmul(pcloud,u), np.matmul(pcloud,v), np.matmul(pcloud,w)
    inside_box = np.where(
                            (pu <= np.dot(P[5],u))&(pu >= np.dot(P[1],u))\
                           &(pv <= np.dot(P[3],v))&(pv >= np.dot(P[2],v))\
                           &(pw <= np.dot(P[3],w))&(pw >= np.dot(P[1],w))
                         )[0]
    return inside_box



def fit_line(x1,x2,y1,y2):
    """Return the coefficients a,b in y(x)=a*x+b, given two points"""
    a = (y2-y1)/(x2-x1)
    b = (x2*y1 - x1*y2)/(x2-x1)
    return a,b

def fit_parabola(x1,x2,x3,y1,y2,y3):
    """Returns the parabola coefficients a,b,c given 3 data points [y(x)=a*x**2+b*x+c]"""
    denom = (x1-x2)*(x1-x3)*(x2-x3)
    a = (x3*(y2-y1)+x2*(y1-y3)+x1*(y3-y2))/denom
    b = (x1**2*(y2-y3)+x3**2*(y1-y2)+x2**2*(y3-y1))/denom
    c = (x2**2*(x3*y1-x1*y3)+x2*(x1**2*y3-x3**2*y1)+x1*x3*(x3-x1)*y2)/denom
    return a,b,c

def fit_cubic(p1, p2, d1, d2):
    """ Returns coefficients a,b,c,d [y(x)=ax^3+bx^2+cx+d]
            Args:
                -p1: tuple (x1,y1) for the coordinates of points 1
                -p2: tuple (x2,y2) for the coordinates of points 2
                -d1: derivative at x1
                -d2: derivative at x2
    """
    x1,y1 = p1
    x2,y2 = p2
    Mat_a = np.array([[x1**3, x1**2, x1, 1],
                      [x2**3, x2**2, x2, 1],
                      [3*x1**2, 2*x1, 1, 0],
                      [3*x2**2, 2*x2, 1, 0]])
    Mat_b = np.array([y1, y2, d1, d2])
    return np.linalg.solve(Mat_a, Mat_b)




if __name__ == '__main__':

    #test of points_inside

    box3d = np.array([5,5,5,0.5,0.5,0.5,0.1,0.9,3.14])

    pts = np.vstack([np.array([0,0,0]), np.array([5,5,5])])  

    tf_Lidar_from_Box = tf_from_pos_euler(box3d[0:3], np.rad2deg(box3d[6:9]))

    aabb = np.vstack([-box3d[3:6]/2,box3d[3:6]/2] )

    print(points_inside_box_mask(pts, aabb, tf_inv(tf_Lidar_from_Box)))

    

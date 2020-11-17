
from pioneer.common import plane, linalg
from pioneer.common.logging_manager import LoggingManager

from numpy.matlib import repmat

import math
import numpy as np
import os
import transforms3d

def grid(v, h, v_from, v_to, h_from, h_to, dtype = np.float32):
    '''
    Computes a matrix of all possible pairs of angles in the 2d field of view.

    \param v vertical resolution
    \param h horizontal resolution
    \param v_fov vertical field of view (degrees)
    \param h_fov horizontal field of view (degrees)
    \param dtype the numpy data type
    '''
    # If you are wondering why there is a complex number here. Read the
    # numpy mgrid documentation. The lines below are equivalent to:
    # a = np.linspace(v_from, v_to, v)
    # b = np.linspace(h_from, h_to, h)
    # b, a = np.meshgrid(b, a)
    a, b = np.mgrid[  v_from:v_to:complex(0,v)
                    , h_from:h_to:complex(0,h)]
    return np.c_[a.ravel(), b.ravel()].astype(dtype)

def from_specs_dict(specs):
    return (specs[k] for k in ['v', 'h', 'v_fov', 'h_fov'])

def angles(v, h = None, v_fov = None, h_fov = None, dtype = np.float32):
    '''
    Computes a matrix of all possible pairs of angles in the 2d field of view.

    The generated grid follows the LCA axis system convention. That is the
    bottom-left corner (0, 0) corresponds to (-v_fov/2, -h_fov/2) and the top
    right corner (v-1, h-1) is (+v_fov/2, +h_fov/2). The grid is generated in
    a row-major order.

    \param v vertical resolution (or a dict with keys v', 'h', 'v_fov', 'h_fov')
    \param h horizontal resolution
    \param v_fov vertical field of view (degrees)
    \param h_fov horizontal field of view (degrees)
    \param dtype the numpy data type
    '''

    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    v_fov_rad = math.radians(v_fov)
    h_fov_rad = math.radians(h_fov)

    v_offset = v_fov_rad/v/2
    h_offset = h_fov_rad/h/2

    return grid(v,h, -v_fov_rad/2 + v_offset, v_fov_rad/2 - v_offset
                   , -h_fov_rad/2 + h_offset, h_fov_rad/2 - h_offset, dtype)

def raycast_angles(v, h = None, v_fov = None, h_fov = None, density = 10, dtype = np.float32):
    '''
    Computes a densified matrix of all possible pairs of angles in the 2d field of view.
    This matrix can be used to cast density * density rays per fov solid angle ('pixel')
    \return the angle grid, and a mapping matrix m, where, m[dense_ray_i] == channel_i
    '''

    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    v_fov_rad = math.radians(v_fov)
    h_fov_rad = math.radians(h_fov)


    dense_to_sparse = np.empty(v*h*density*density, 'u4')
    sparse_to_dense = np.empty((v*h, density, density), 'u4')
    dense_to_sub = np.empty((v*h*density*density, 2), 'u4')
    m_i = 0
    for v_i in range(v):
        for vd_i in range(density):
            for h_i in range(h):
                for hd_i in range(density):
                    sparse_i = v_i * h + h_i
                    dense_to_sparse[m_i] = sparse_i
                    sparse_to_dense[sparse_i, vd_i, hd_i] = m_i
                    dense_to_sub[m_i] = [vd_i, hd_i]
                    m_i += 1

    return grid(v * density,h * density, -v_fov_rad/2,  v_fov_rad/2
                   , -h_fov_rad/2, h_fov_rad/2, dtype), dense_to_sparse, sparse_to_dense, dense_to_sub

def custom_v_angles(v, h = None, v_fov = None, h_fov = None, factor = 1, filename = os.path.join(os.path.dirname(__file__), 'eagle_angles_80.txt'), dtype = np.float32):
    '''
    similar to \a angles() but using a file to define scan direction angles
    '''
    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    h_fov_rad = math.radians(h_fov)
    h_offset = h_fov_rad/h/2
    a = np.genfromtxt(filename, delimiter='\n', converters={_:lambda s: int(s, 16) for _ in range(1)}) 
    a = a[:v]
    a = a/2**16 * v_fov - v_fov/2 
    a = np.deg2rad(a)  * factor 
    b = np.linspace(-h_fov_rad/2 + h_offset, h_fov_rad/2 - h_offset, num = h, dtype = dtype)
    b, a = np.meshgrid(b, a)
    return np.c_[a.ravel(), b.ravel()].astype(dtype)

def custom_v_quad_directions(v, h = None, v_fov = None, h_fov = None, factor = 1, filename = os.path.join(os.path.dirname(__file__), 'eagle_angles_80.txt'), dtype = np.float32):
    '''
    similar to \a quad_directions() but using a file to define scan direction angles
    '''

    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    v_fov_rad = math.radians(v_fov)
    h_fov_rad = math.radians(h_fov)

    v_cell_size = v_fov_rad/v
    h_cell_size = h_fov_rad/h

    file_angles = np.genfromtxt(filename, delimiter='\n', converters={_:lambda s: int(s, 16) for _ in range(1)}) 

    def custom_grid(v, h, v_offset, h_offset_from, h_offset_to, dtype):
        a = file_angles[:v]
        a = a/2**16 * v_fov - v_fov/2
        a = (np.radians(a) - v_offset)  * factor    
        b = np.linspace(-h_fov_rad/2+h_offset_from, h_fov_rad/2+h_offset_to, num = h, dtype = dtype)
        b, a = np.meshgrid(b, a)

        return np.c_[a.ravel(), b.ravel()].astype(dtype)
    
    return np.vstack((
     directions(custom_grid(v,h,-v_cell_size/2 ,h_cell_size , 0           , dtype))
    ,directions(custom_grid(v,h,+v_cell_size/2 ,h_cell_size , 0           , dtype))
    ,directions(custom_grid(v,h,+v_cell_size/2 ,0           , -h_cell_size, dtype))
    ,directions(custom_grid(v,h,-v_cell_size/2 ,0           , -h_cell_size, dtype)))
    )

def direction(theta_x, theta_y):
    '''
    Convert angles of a spherical axis sytem into a cartesian direction vector.
    The cartesian axis system is the camera axis system.

      z

      +-------> x
      |
      |
    y v

    The z axis enters your screen (or paper if you are the kind of person that
    still prints code).

    Angles go from -fov/2 to fov/2 in both horizontal and vertical direction, always computed
    using "right hand" convention. In each direction, maximum z component will be attained at angle 0.

    In the x-z plane (viewed from above):

                       
                       pi/2
                    x  ^
                       |
                       |<--.
                       |th_y\
          ------------(.)-------------> z
                       y
                       |
                       |
                       -pi/2
    x = sin(theta_y)
    z = cos(theta_y) //we want x,z = (0,1) at theta_y = 0

    In the y-z plane (view from side):


                    z  ^
                       |
                       |<--.
                       |th_x \        y
       pi ------------(.)-------------> 
                       x
                       
    y = cos(theta_x + pi/2)
    z = sin(theta_x + pi/2) //we want (y,z) = (0,1) at theta_x = 0

    So the x, y, z coordinates should follow the equations below

    x = sin(theta_y)
    y = cos(theta_x + pi/2)
    z = cos(theta_y) * sin(theta_x + pi/2)
    '''
    x = np.sin(theta_y)
    y = np.cos(theta_x + np.pi/2)
    z = np.sin(theta_x + np.pi/2) * np.cos(theta_y)
    return x, y, z

def direction_spherical(thetas_x, thetas_y):
    '''
    LeddarConfig implementation
    '''
    x = np.cos(thetas_x) * np.sin(thetas_y)
    y = np.sin(thetas_x)
    z = np.cos(thetas_x) * np.cos(thetas_y)
    return x, y, z

def direction_orthogonal(thetas_x, thetas_y):
    '''
    Simulator implementation using orthogonal camera depth projection
    '''

    x = np.tan(thetas_y) 
    y = np.tan(-thetas_x)
    z = np.ones_like(x)

    n = np.sqrt(z**2 + x**2 + y**2)

    return x/n, y/n, z/n


def directions(angles, direction_f = direction):
    '''Generate a set of cartesian direction vectors from a grid of
    spherical coordinates angles. This function uses the same convention as
    the `direction` function.
    '''
    thetas_x, thetas_y = angles.T

    return np.stack(direction_f(thetas_x, thetas_y), axis=1)

def directions_orthogonal(v,h=None,v_fov=None,h_fov=None, dtype = np.float32):
    '''Generate a set of cartesian direction vectors from a grid of
    2D pixels coordinates (eg : camera depth) using  Carla Simulator implementation 
    and camera depth projection
    '''                  
    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    if h_fov > 90:
        LoggingManager.instance().warning("The projection model is not adapted for horizontal fov greater than 90 degrees. Trying to correct the" \
                     +" situation by spliting the fov in three parts and re-merging them. Use 'projection: direction_carla_pixell' instead.")
        return directions_orthogonal_pixell(v=v, h=h, v_fov=v_fov, h_fov=h_fov, dtype=dtype)

    # (Intrinsic) K Matrix
    k = np.identity(3)
    k[0, 2] = h / 2.0
    k[1, 2] = v / 2.0
    k[0, 0] = k[1, 1] = h / \
        (2.0 * math.tan(h_fov * math.pi / 360.0))
    # 2d pixel coordinates
    pixel_length = h * v
    u_coord = repmat(np.r_[h-1:-1:-1],
                     v, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[v-1:-1:-1],
                     1, h).reshape(pixel_length)
    # pd2 = [u,v,1]
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
    direction = np.dot(np.linalg.inv(k), p2d).T
    direction[:,0] = -direction[:,0]

    v_cell_size, h_cell_size = v_h_cell_size_rad(v, h, v_fov, h_fov)

    # First face
    face_a = np.zeros((direction.shape))
    face_a[:,0] = direction[:,0] - h_cell_size/2
    face_a[:,1] = direction[:,1] - v_cell_size/2
    face_a[:,2] = direction[:,2]

    # Second face
    face_b = np.zeros((direction.shape))
    face_b[:,0] = direction[:,0] + h_cell_size/2
    face_b[:,1] = direction[:,1] - v_cell_size/2
    face_b[:,2] = direction[:,2]

    # Third face
    face_c = np.zeros((direction.shape))
    face_c[:,0] = direction[:,0] + h_cell_size/2
    face_c[:,1] = direction[:,1] + v_cell_size/2
    face_c[:,2] = direction[:,2]
    
    # Fourth face
    face_d = np.zeros((direction.shape))
    face_d[:,0] = direction[:,0] - h_cell_size/2
    face_d[:,1] = direction[:,1] + v_cell_size/2
    face_d[:,2] = direction[:,2]

    quad_direction = np.vstack((face_a,face_b,face_c,face_d))

    return direction,quad_direction


def directions_orthogonal_pixell(v, h=None, v_fov=None, h_fov=None, dtype = np.float32):
    """Returns directions and quad_directions for the carla simulator projection, in the case of a h_fov greater than 90 deg."""

    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    directions_central_third, quad_directions_central_third = directions_orthogonal(v=v, h=int(h/3), v_fov=v_fov, h_fov=h_fov/3)

    rot_left = transforms3d.euler.euler2mat(0,np.deg2rad(h_fov/3),0)
    rot_right = transforms3d.euler.euler2mat(0,np.deg2rad(-h_fov/3),0)

    directions_left_third = directions_central_third @ rot_left
    directions_right_third = directions_central_third @ rot_right
    quad_directions_left_third = quad_directions_central_third @ rot_left
    quad_directions_right_third = quad_directions_central_third @ rot_right

    ind_tpm = np.arange(v*int(h/3)).reshape((v,int(h/3)))
    ind = np.ravel(np.hstack([ind_tpm,ind_tpm+v*int(h/3),ind_tpm+2*v*int(h/3)]))
    quad_ind_tpm = np.arange(4*v*int(h/3)).reshape((4*v,int(h/3)))
    quad_ind = np.ravel(np.hstack([quad_ind_tpm,quad_ind_tpm+4*v*int(h/3),quad_ind_tpm+2*4*v*int(h/3)]))

    directions = np.vstack([directions_left_third, directions_central_third, directions_right_third])[ind]
    quad_directions = np.vstack([quad_directions_left_third, quad_directions_central_third, quad_directions_right_third])[quad_ind]

    return directions, quad_directions
                   

def v_h_cell_size_rad(v, h = None, v_fov = None, h_fov = None, output_fov = False):
    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    v_fov_rad = math.radians(v_fov)
    h_fov_rad = math.radians(h_fov)

    v_cell_size = v_fov_rad/v
    h_cell_size = h_fov_rad/h
    
    if output_fov:
        return v_cell_size, h_cell_size, v_fov_rad, h_fov_rad
    else:
        return v_cell_size, h_cell_size


def quad_angles(v, h=None, v_fov=None, h_fov=None, 
                dtype=np.float32):
    """ Like angles(), but for quad-stuff. 
    """
    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    v_cell_size, h_cell_size, v_fov_rad, h_fov_rad = v_h_cell_size_rad(v, h, v_fov, h_fov, True)
    return np.vstack((
     grid(v, h, -v_fov_rad/2, v_fov_rad/2-v_cell_size , -h_fov_rad/2, h_fov_rad/2-h_cell_size, dtype)
    ,grid(v, h, -v_fov_rad/2+v_cell_size, v_fov_rad/2, -h_fov_rad/2, h_fov_rad/2-h_cell_size, dtype)
    ,grid(v, h, -v_fov_rad/2+v_cell_size, v_fov_rad/2, -h_fov_rad/2+h_cell_size, h_fov_rad/2, dtype)
    ,grid(v, h, -v_fov_rad/2, v_fov_rad/2-v_cell_size, -h_fov_rad/2+h_cell_size, h_fov_rad/2, dtype)
    ))


def custom_quad_angles(specs,angles, dtype = np.float32):
    v, h, v_fov, h_fov = from_specs_dict(specs)

    v_fov_rad = math.radians(v_fov)
    h_fov_rad = math.radians(h_fov)

    v_cell_size = v_fov_rad/v
    h_cell_size = h_fov_rad/h

    def custom_grid(v_offset, h_offset, dtype):
        a = angles[:,0] - v_offset    
        b = angles[:,1] - h_offset

        return np.c_[a.ravel(), b.ravel()].astype(dtype)
    
    return np.vstack((
     custom_grid(-v_cell_size/2 ,-h_cell_size/2 , dtype)
    ,custom_grid(+v_cell_size/2 ,-h_cell_size/2 , dtype)
    ,custom_grid(+v_cell_size/2 ,+h_cell_size/2 , dtype)
    ,custom_grid(-v_cell_size/2 ,+h_cell_size/2 , dtype))
    )


def quad_directions(v, h = None, v_fov = None, h_fov = None, dtype = np.float32, direction_f = direction):

    if isinstance(v, dict):
        v, h, v_fov, h_fov = from_specs_dict(v)

    v_cell_size, h_cell_size, v_fov_rad, h_fov_rad = v_h_cell_size_rad(v, h, v_fov, h_fov, True)

    return np.vstack((
     directions(grid(v,h,-v_fov_rad/2              ,v_fov_rad/2-v_cell_size ,-h_fov_rad/2             , h_fov_rad/2-h_cell_size , dtype), direction_f = direction_f)
    ,directions(grid(v,h,-v_fov_rad/2+v_cell_size  ,v_fov_rad/2             ,-h_fov_rad/2             , h_fov_rad/2-h_cell_size , dtype), direction_f = direction_f)
    ,directions(grid(v,h,-v_fov_rad/2+v_cell_size  ,v_fov_rad/2             ,-h_fov_rad/2+h_cell_size , h_fov_rad/2             , dtype), direction_f = direction_f)
    ,directions(grid(v,h,-v_fov_rad/2              ,v_fov_rad/2-v_cell_size ,-h_fov_rad/2+h_cell_size , h_fov_rad/2             , dtype), direction_f = direction_f))
    )

def frustrum_old(v_fov, h_fov, scale, dtype = np.float32, direction_f = direction):
    v_fov_rad_2 = math.radians(v_fov)/2
    h_fov_rad_2 = math.radians(h_fov)/2

    d = [direction_f(-v_fov_rad_2, -h_fov_rad_2)
    , direction_f(-v_fov_rad_2, h_fov_rad_2)
    , direction_f(v_fov_rad_2, -h_fov_rad_2)
    , direction_f(v_fov_rad_2, h_fov_rad_2)]

    vertices = np.empty((5, 3), dtype)

    vertices[0] = [0,0,0]

    vertices[1:] = np.array(d, dtype) * scale

    indices = np.array([ 0,1 , 0,2 , 0,3 , 0,4 , 1,2 , 2,4 , 4,3, 3,1], dtype = "uint32")

    return indices, vertices


def frustrum_directions(v_fov, h_fov, dtype = np.float32, direction_f = direction):
    v_fov_rad_2 = math.radians(v_fov)/2
    h_fov_rad_2 = math.radians(h_fov)/2

    return  np.array([direction_f(-v_fov_rad_2, -h_fov_rad_2)
    , direction_f(-v_fov_rad_2, h_fov_rad_2)
    , direction_f(v_fov_rad_2, h_fov_rad_2)
    , direction_f(v_fov_rad_2, -h_fov_rad_2)], dtype)


def custom_frustrum_directions(custom_v_angles, v_cell_size, h_cell_size, dtype = np.float32):

    min_v, max_v = custom_v_angles[:,0].min(), custom_v_angles[:,0].max()
    min_h, max_h = custom_v_angles[:,1].min(), custom_v_angles[:,1].max()

    return  np.array([
        direction(min_v - v_cell_size/2, min_h - h_cell_size/2)
    ,   direction(min_v - v_cell_size/2, max_h + h_cell_size/2)
    ,   direction(max_v + v_cell_size/2, max_h + h_cell_size/2)
    ,   direction(max_v + v_cell_size/2, min_h - h_cell_size/2)], dtype)


def frustrum(frustrum_directions, scale = 10):
    
    vertices = np.empty((5, 3), frustrum_directions.dtype)

    vertices[0] = [0,0,0]

    vertices[1:] = frustrum_directions * scale

    indices = np.array([ 0,1 , 0,2 , 0,3 , 0,4 , 1,2 , 2,3 , 3,4, 4,1], dtype = 'u4')

    return indices, vertices

def frustrum_planes(frustrum_directions):
    d = frustrum_directions

    return np.vstack((plane.make_plane([0,0,0], linalg.normalized(np.cross(d[0], d[1])), d.dtype), 
    plane.make_plane([0,0,0], linalg.normalized(np.cross(d[1], d[2])), d.dtype),
    plane.make_plane([0,0,0], linalg.normalized(np.cross(d[2], d[3])), d.dtype),
    plane.make_plane([0,0,0], linalg.normalized(np.cross(d[3], d[0])), d.dtype)))


def to_point_cloud(selection, distances, directions, dtype = np.float32):

    sn = selection.shape[0]

    points = np.empty((sn, 3), dtype)

    points.resize((sn, 3))
    points[:] = directions[selection] * distances.reshape(sn, 1)

    return points

def generate_quads_indices(n, dtype = np.uint32):

    iota = np.arange(n, dtype = dtype)
    iota_2n = iota+2*n
    return np.stack((iota, iota_2n, iota+n, iota, iota + 3*n, iota_2n), axis=1)

def triangle_to_echo_index(triangle):
    return triangle[0] # must be in accordance with generate_quads_indices()

def quad_stack(scalars):
    return np.concatenate((scalars,scalars,scalars,scalars))

def to_quad_cloud(selection, distances, amplitudes, quad_directions, v, h,  dtype = np.float32):

    sn = selection.shape[0]
    
    n = v * h
    
    points = np.empty((sn*4, 3), dtype)

    quad_amplitudes = np.empty((sn*4, 1), dtype)

    # four points per quad, 1 different direction per point, same distance for each
    points[0:sn]      = quad_directions[selection    ] * distances[:, np.newaxis]
    points[sn:2*sn]   = quad_directions[selection+n  ] * distances[:, np.newaxis]
    points[2*sn:3*sn] = quad_directions[selection+2*n] * distances[:, np.newaxis]
    points[3*sn:]     = quad_directions[selection+3*n] * distances[:, np.newaxis]

    # same amplitude for each four points
    quad_amplitudes[:] = quad_stack(amplitudes)[:, np.newaxis]
    
    # a quad is formed with 2 triangles
    quad_indices = generate_quads_indices(sn, np.uint32)

    return points, quad_amplitudes, quad_indices.flatten()

def convert_echo_package(old, specs = {"v" : None, "h" : None, "v_fov" : None, "h_fov" : None}):
    return to_echo_package(old['indices']
                           , distances = old['data'][:,1]
                           , amplitudes = old['data'][:,2]
                           , timestamps = old['data'][:,0].astype('u2')
                           , flags = old['flags'].astype('u2')
                           , timestamp = old['timestamp'] if 'timestamp' in old else old['data'][0,0]
                    , specs = specs
                    , distance_scale = 1.0, amplitude_scale = 1.0, led_power = 1.0)

def to_echo_package(indices = np.array([], 'u4'), distances = np.array([], 'f4'), amplitudes = np.array([], 'f4')
                    , timestamps = None, flags = None, timestamp = 0
                    , specs = {"v" : None, "h" : None, "v_fov" : None, "h_fov" : None}
                    , distance_scale = 1.0, amplitude_scale = 1.0, led_power = 1.0, eof_timestamp = None
                    , additionnal_fields = {}):
    '''
        This format MUST remain in synch with the one in LeddarPyDevice::PackageEchoes()
        additionnal_fields format example : {'widths':[np.array([values]), 'f4']}
    '''
    package = specs.copy()

    if eof_timestamp is None:
        eof_timestamp = timestamp

    package.update({"timestamp": timestamp
                   , "eof_timestamp": eof_timestamp
                   , "distance_scale": distance_scale
                   , "amplitude_scale": amplitude_scale
                   , "led_power": led_power})

    assert(indices.size == distances.size == amplitudes.size)

    if timestamps is not None:
        assert(indices.size == timestamps.size)
    if flags is not None:
        assert(indices.size == flags.size)

    default_fields = [('indices', 'u4')
                    , ('distances', 'f4')
                    , ('amplitudes', 'f4')
                    , ('timestamps', 'u8')
                    , ('flags', 'u2')]

    dtype = np.dtype(default_fields + [(key,additionnal_fields[key][1]) for key in additionnal_fields.keys()])

    package['data'] = np.empty(indices.size, dtype = dtype)

    package['data']["indices"] = indices
    package['data']["distances"] = distances
    package['data']["amplitudes"] = amplitudes
    package['data']["timestamps"] = 0 if timestamps is None else timestamps
    package['data']["flags"] = 1 if flags is None else flags

    for key in additionnal_fields.keys():
        package['data'][key] = additionnal_fields[key][0]

    return package

def to_traces_package(traces, start_index = 0, timestamp = 0):
    return dict(data=traces, start_index=start_index, timestamp=timestamp)


def echo_package_to_point_cloud(package):
    """ Return all the points and corresponding amplitudes from an echo package in one step.
    """
    theta = angles(package['v'], package['h'], package['v_fov'], package['h_fov'], dtype = np.float32)
    vec = directions(theta)
    X = to_point_cloud(package['data']["indices"], package['data']["distances"], vec, dtype = np.float32)
    return X, package['data']["amplitudes"]
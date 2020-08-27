import numpy as np

'''
Plane API
a plane is defined by ax + by + cz + d = 0
plane := np.array([a,b,c,d], dtype = ...)
'''

def make_plane(point, direction, dtype = np.float32):
    p =  np.zeros((4,), dtype)
    p[0:3] = direction
    p[3] = -np.dot(direction, point)
    return p

def plane_normal(plane):
    return plane[0:3]

def plane_offset(plane):
    return -plane[3]

def plane_point(plane):
    return plane_normal(plane) * plane_offset(plane)

def plane_test(plane, points):
    return np.dot(plane_normal(plane), points.T) <= plane[3]

def intersect_rays(points, directions, plane):
    n = plane_normal(plane)
    dens = np.dot(directions, n)

    if np.any(dens < 1e-6):
        raise RuntimeError("Ray and plane are parallel or almost parallel")

    return np.dot(plane_point(plane) - points, n) / dens

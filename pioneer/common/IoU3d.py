from shapely.geometry import Polygon

import numpy as np

def _tf_rot2d(theta, pt):
        _c, _s = np.cos(theta), np.sin(theta)
        R = np.array(((_c, -_s), (_s, _c)))
        return np.dot(R, pt.reshape((2, 1)))

def IoU3drot(Z0, Z1):
    """ Compute the IoU between two boxes in 3d, with one angle along a given axis
            Z0, Z1: [c,l,r,rad_r], 

            c - is the center of the boxe
            l - is the size of the boxe
            r - axis or rotation, must be one of {'x','y','z'}, must be the same for Z0 and Z1
            rad_r - angle of rotation, in radian, in the natural convention (counter-clockwise)


            for example:  c = np.array([10.1,3.2,2.0]), l = np.array([10.1,3.2,2.0]), r = 'x', rad_r = np.pi/6.0

            r is one of {'x','y','z'}, both boxes must have the same
    """

    c0, l0, theta0 = Z0[0], Z0[1], Z0[3]
    c1, l1, theta1 = Z1[0], Z1[1], Z1[3]

    distance = ((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2 + (c0[2]-c1[2])**2)**0.5
    if distance > (max(l0)+max(l1))/2**0.5:
        return 0

    if Z0[2] == 'x':
        i, j, k = 0, 1, 2
    if Z0[2] == 'y':
        i, j, k = 1, 2, 0
    if Z0[2] == 'z':
        i, j, k = 2, 0, 1

    l_i = np.minimum(c0[i]+0.5*l0[i], c1[i]+0.5*l1[i]) - \
        np.maximum(c0[i]-0.5*l0[i], c1[i]-0.5*l1[i])

    l_i = np.maximum(l_i, 0)

    if l_i > 0:
        pts0 = [np.array([-0.5*l0[j], 0.5*l0[k]]), np.array([0.5*l0[j], 0.5*l0[k]]),
                np.array([0.5*l0[j], -0.5*l0[k]]), np.array([-0.5*l0[j], -0.5*l0[k]])]
        pts1 = [np.array([-0.5*l1[j], 0.5*l1[k]]), np.array([0.5*l1[j], 0.5*l1[k]]),
                np.array([0.5*l1[j], -0.5*l1[k]]), np.array([-0.5*l1[j], -0.5*l1[k]])]

        polyg0 = Polygon([np.array([c0[j], c0[k]]) +
                          _tf_rot2d(theta0, _p).reshape((2)) for _p in pts0])
        polyg1 = Polygon([np.array([c1[j], c1[k]]) +
                          _tf_rot2d(theta1, _p).reshape((2)) for _p in pts1])

        intersection = polyg0.intersection(polyg1).area * l_i
    
        Vol0, Vol1 = np.prod(l0), np.prod(l1)

        return (intersection)/(Vol0 + Vol1 - intersection + 1e-9)

    else:
        return 0


def matrixIoU(Z0, Z1=None):
    """
    compute the IoU matrix between the two sets of 3d bboxes with one angle.
            Z0, Z1: [c,l,r,rad_r]
                            c - (M,3)
                            l - (M,3)
                            r - one of 'x', 'y', 'z'
                            rad_r - (M)

            If Z1 is None, then the matrix is between Z0 and itself.
    """
    r = Z0[2]
    if Z1 is None:  # (symetric case)
        M = Z0[0].shape[0]
        matiou = np.zeros((M, M), dtype=np.float)

        for m in range(M):
            for n in range(m+1, M):
                matiou[m, n] = IoU3drot([Z0[0][m, :], Z0[1][m, :], r, Z0[3][m]], [
                                        Z0[0][n, :], Z0[1][n, :], r, Z0[3][n]])

        return matiou + matiou.T + np.eye(M)

    else:
        M, N = Z0[0].shape[0], Z1[0].shape[0]
        matiou = np.zeros((M, N), dtype=np.float)

        for m in range(M):
            for n in range(N):

                matiou[m, n] = IoU3drot([Z0[0][m, :], Z0[1][m, :], r, Z0[3][m]], [
                                        Z1[0][n, :], Z1[1][n, :], r, Z1[3][n]])

        
        return matiou




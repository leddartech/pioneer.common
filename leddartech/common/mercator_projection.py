import cv2
import numpy as np

class MercatorProjection():
    
    '''Mercator projection for fisheye lens camera images.
    '''
    def __init__(self,
                    intrinsic_matrix,
                    distortion_coeffs,
                    config={  
                        'radius':1.0, 
                        'fov_h':np.deg2rad(180), 
                        'fov_v':np.deg2rad(2*75),
                        'center_azimuth':0.0, 
                        'image_h':1440, 
                        'image_v':1080
                        }):
        
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.radius = config.get('radius', 1.0)
        self.fov_h = config.get('fov_h', np.deg2rad(180))
        self.fov_v = config.get('fov_v', np.deg2rad(2*75))
        self.center_azimuth = config.get('center_azimuth', 0.0)
        self.image_h = config.get('image_h', 1440)
        self.image_v = config.get('image_v', 1080)
        
        self._img_pts, self._img_mask = self.__get_grid_sample_image_points()


    def forward(self, theta, phi, radius=1, theta_0=0):

        '''Explode the sphere, i.e. from sphere to plane.

            typically, phi is elevation and theta is azimuth
        '''
        x = radius*(theta-theta_0)
        y = radius*np.log(np.tan(np.pi/4.0 + phi/2.0))
        return x, y

    def backward(self, x, y, radius=1, theta_0=0):

        '''Inverse of forward, from thet Mercator projection plane back to the sphere.
        '''
        theta = theta_0 + x/radius
        phi = 2.0*np.arctan(np.exp(y/radius))-np.pi/2.0
        return theta, phi
    
    def __get_grid_sample_image_points(self):
        x0, y0 = self.forward(self.fov_h/2.0, self.fov_v/2.0, self.radius, self.center_azimuth)
        y, x = np.mgrid[-y0:y0:(2*y0)/(self.image_v), -x0:x0:(2*x0)/(self.image_h)]
        y = y.ravel()
        x = x.ravel()
        theta, phi = self.backward(x,y,self.radius, self.center_azimuth)

        pts_3d = np.vstack([
                    np.cos(phi)*np.sin(theta),
                    np.sin(phi),
                    np.cos(phi)*np.cos(theta),
                ]).T
        pts_3d = pts_3d.reshape((-1,1,3))
        image_pts, _ = cv2.fisheye.projectPoints(pts_3d, 
                                                    np.zeros((3,1)), 
                                                    np.zeros((3,1)), 
                                                    self.intrinsic_matrix, 
                                                    self.distortion_coeffs)
        image_pts = np.squeeze(image_pts)
        image_pts = np.round(image_pts).astype(np.int)
        keep = (image_pts[:,0]>=0) * (image_pts[:,0]<self.image_h) * (image_pts[:,1]>=0) * (image_pts[:,1]<self.image_v)

        return image_pts[keep], keep
    
    def undistort(self, image_distorted):
        undistorded = np.zeros((self.image_h*self.image_v, 3), dtype=np.uint8)
        undistorded[self._img_mask, :] = image_distorted[self._img_pts[:,1], self._img_pts[:,0],:]    
        undistorded = undistorded.reshape((self.image_v, self.image_h,3))
        return undistorded
    
    def project_pts(self, pts):
        x0, y0 = self.forward(self.fov_h/2.0, self.fov_v/2.0, self.radius, self.center_azimuth)
        R = np.linalg.norm(pts, axis=1)
        r = (pts[:,0]**2 + pts[:,2]**2)**0.5
        phi, theta = np.arcsin(pts[:,1]/R), np.arcsin(pts[:,0]/r)
        x,y = self.forward(theta, phi, self.radius, self.center_azimuth)
        y = (y+y0)/(2*y0)*self.image_v
        x = (x+x0)/(2*x0)*self.image_h
        
        pts = np.column_stack((x,y))
        return pts

        

    

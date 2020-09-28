from pioneer.common import linalg

from enum import Enum

import cv2
import numpy as np

class Pos(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class CylindricalProjection():
    ''' cylindrical projection for 3 cameras
    
        args: 
            intrinsic_calibrations: list of the 3 intrinsic cameras calibration
            distortion_coef: list of the 3 distorsion cameras coef
            extrinsic_calibrations: list of the 3 extrinsics 4x4 matrix, the middle is equal to identity
            config: configuration dict
                radius: cylinder radius (meter)
                FOV_h: total horizontal cylinder FOV (rad)
                FOV_v: total vertical cylinder FOV (rad)
                image_h : horizontal cylinder (pixel)
                image_v : vertical cylinder (pixel)
                fusion_overlap_ratio : overlap ratio between take in account for image merging or fusion (0.0 to 1.0)
    '''
    def __init__(self
        , intrinsic_calibrations = None
        , distortion_coef = None
        , extrinsic_calibrations = None
        , config={'radius':50.0, 'FOV_h':np.deg2rad(210), 'FOV_v':np.deg2rad(67.5), 'image_h':2000, 'image_v':int(2000*0.25), 'fusion_overlap_ratio': 0.25}
        ):

        self.__assert_intrinsic_calibrations(intrinsic_calibrations)
        self.__assert_distortion_coefficients(distortion_coef)
        self.__assert_extrinsic_calibrations(extrinsic_calibrations)

        self.radius = config.get('radius', 50.0)
        self.FOV_width = config.get('FOV_h', np.deg2rad(210))
        self.FOV_height = config.get('FOV_v', np.deg2rad(67.5))
        self.image_width = config.get('image_h', 2000)
        self.image_height = config.get('image_v', int(2000*0.25))
        self.fusion_overlap_ratio = config.get('fusion_overlap_ratio', 0.25)

        self.cylinder_points, self.cylinder_points_2d = self.__get_cylinder_points(self.image_width, self.image_height, self.FOV_width, self.FOV_height, self.radius)

        self.intrinsic_calibrations = {}
        self.extrinsic_calibrations = {}
        self.distortion_coefficients = {}
        self.new_matrices = {}
        self.keeped_in_cam_points = {}
        self.keeped_cylinder_points_2d = {}
        self.images_min_x = {}
        self.images_max_x = {}
        self.masks = {}

        for pos in Pos:
            self.intrinsic_calibrations[pos] = intrinsic_calibrations[pos.value]
            self.distortion_coefficients[pos] = distortion_coef[pos.value]
            self.extrinsic_calibrations[pos] = extrinsic_calibrations[pos.value]
            self.new_matrices[pos] = None
            self.keeped_in_cam_points[pos] = None
            self.keeped_cylinder_points_2d[pos] = None
            self.images_min_x[pos] = None
            self.images_max_x[pos] = None
            self.masks[pos] = None

    def project_pts(self, pts, mask_fov=False, output_mask=False, margin=0):
        ''' project 3D in the 2D cylindrical referiencial

            Args:
                pts_3D: 3D point in the center camera referential (3xN)
                mask_fov (optionnal): removes points outside the fov
                output_mask (optionnal): if True, returns the mask applied to the points
                margin (optionnal): margin (in pixels) outside the image unaffected by the fov mask
            Return:
                2xN: 2D points in cylindrical image referential
                mask (optionnal): returned if output_mask is True
        '''
        assert len(pts.shape)==2, '2 dimensionals array needed'
        assert pts.shape[0]==3, '3d points format 3xN'

        azimut = np.arctan2(pts[0,:], pts[2,:])
        norm_xz = np.linalg.norm(pts[[0,2],:], axis = 0)
        elevation = np.arctan2(pts[1,:], norm_xz)

        x = (self.image_width/2 + azimut * (self.image_width/self.FOV_width)).astype(int)
        y = (self.image_height/2 + elevation * (self.image_height/self.FOV_height)).astype(int)

        pts = np.column_stack((x,y))

        mask = (pts[2,:] > 0)
        if mask_fov or output_mask:
            mask = (azimut > -self.FOV_width/2 - margin/self.image_width*self.FOV_width) & \
                   (azimut < self.FOV_width/2 + margin/self.image_width*self.FOV_width) & \
                   (elevation > -self.FOV_height/2 - margin/self.image_height*self.FOV_height) & \
                   (elevation < self.FOV_height/2 + margin/self.image_height*self.FOV_height)
        if mask_fov:
            pts = pts[mask]

        if output_mask:
            return pts, mask
        return pts

    def stitch(self, images=None):
        self.__assert_image(images)

        for i, image in enumerate(images):
            if image.ndim == 2:
                images[i] = self.gray_to_rgb(image)

        rectified_images = dict()
        cylinder = dict()
        
       
        for position in Pos:
            # only the first time, compute matrix able to reproject in the undistord image
            if self.new_matrices[position] is None:
                self.new_matrices[position] = self.__compute_optimal_new_matrix(images[position.value], self.intrinsic_calibrations[position], self.distortion_coefficients[position])
            # undistor the image
            rectified_images[position] = cv2.undistort(images[position.value], self.intrinsic_calibrations[position], self.distortion_coefficients[position], None, self.new_matrices[position])
            # each camera will be reprojected in these cylinder images
            cylinder[position] = np.zeros([self.image_height, self.image_width, 3], dtype=images[Pos.CENTER.value].dtype)
        
        # only the first time, compute LUT for each camera to project 2D camera image in the cylinder
        if self.__are_keeped_in_cam_points_none():
            self.__compute_lookup_table_cameras_to_cylinders(images[Pos.CENTER.value])
        # only the first time, compute masks for each camera used to merge them in the final cylinder image
        if self.__are_masks_none():
            self.__compute_mask()
 
        # do the projection in each cylinder
        for position in Pos:
            cylinder[position][self.keeped_cylinder_points_2d[position][1,:],self.keeped_cylinder_points_2d[position][0,:],:] = rectified_images[position][self.keeped_in_cam_points[position][1,:],self.keeped_in_cam_points[position][0,:],:]
        
        # nerge the 3 projected image in a final cylinder image
        pano = cylinder[Pos.LEFT] * np.tile(self.masks[Pos.LEFT][:,:,np.newaxis],3) + cylinder[Pos.CENTER] * np.tile(self.masks[Pos.CENTER][:,:,np.newaxis],3) + cylinder[Pos.RIGHT] * np.tile(self.masks[Pos.RIGHT][:,:,np.newaxis],3)
        return pano.astype(np.uint8)

    def __compute_optimal_new_matrix(self, image, matrix, distortion_coefficient):
        height, width = image.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion_coefficient, (width,height), 0, (width,height))
        return new_camera_matrix

    #TODO: Refactor this function
    def __compute_lookup_table_cameras_to_cylinders(self, image_center):
        # u,v,scale in camera, used to checked if points are in front or behind the camera
        pt_in_cam_3 = self.new_matrices[Pos.CENTER] @ self.cylinder_points[:3,:]
        # project 3D cylinder points in 2D image
        pt_in_cam = (cv2.projectPoints(self.cylinder_points[:3,:],np.zeros((3, 1)),np.zeros((3, 1)), self.new_matrices[Pos.CENTER], self.distortion_coefficients[Pos.CENTER]*0.0))[0].reshape(-1,2).T
        # keep point respect image shape, and point in front of the camera
        keep = np.logical_and(np.logical_and(np.logical_and(np.logical_and(pt_in_cam[0,:] >=0, pt_in_cam[0,:] < image_center.shape[1]), pt_in_cam[1,:] >=0), pt_in_cam[1,:] < image_center.shape[0] ),pt_in_cam_3[2,:]>0)
        self.keeped_in_cam_points[Pos.CENTER] = pt_in_cam[:,keep].astype(np.int)
        self.keeped_cylinder_points_2d[Pos.CENTER] = self.cylinder_points_2d[:,keep].astype(np.int)
        # compute left and right image limits in the cylinder, used to creat the right merging masks
        self.images_min_x[Pos.CENTER] = self.keeped_cylinder_points_2d[Pos.CENTER][0,self.keeped_cylinder_points_2d[Pos.CENTER].reshape(2,-1)[1,:]==self.image_height//2].min()
        self.images_max_x[Pos.CENTER] = self.keeped_cylinder_points_2d[Pos.CENTER][0,self.keeped_cylinder_points_2d[Pos.CENTER].reshape(2,-1)[1,:]==self.image_height//2].max()

        # left camera
        calib_extrinsic_l_c_inv = linalg.tf_inv(self.extrinsic_calibrations[Pos.LEFT])
        pt_in_cam_3 = self.new_matrices[Pos.LEFT] @ (calib_extrinsic_l_c_inv @ self.cylinder_points)[:3,:]
        pt_in_cam_3d = (calib_extrinsic_l_c_inv @ self.cylinder_points)[:3,:]
        pt_in_cam = (cv2.projectPoints(pt_in_cam_3d,np.zeros((3, 1)),np.zeros((3, 1)), self.new_matrices[Pos.LEFT], self.distortion_coefficients[Pos.LEFT]*0.0))[0].reshape(-1,2).T
        keep = np.logical_and(np.logical_and(np.logical_and(np.logical_and(pt_in_cam[0,:] >=0, pt_in_cam[0,:] < image_center.shape[1]), pt_in_cam[1,:] >=0), pt_in_cam[1,:] < image_center.shape[0] ),pt_in_cam_3[2,:]>0)
        self.keeped_in_cam_points[Pos.LEFT] = pt_in_cam[:,keep].astype(np.int)
        self.keeped_cylinder_points_2d[Pos.LEFT] = self.cylinder_points_2d[:,keep].astype(np.int)
        self.images_min_x[Pos.LEFT] = self.keeped_cylinder_points_2d[Pos.LEFT][0,self.keeped_cylinder_points_2d[Pos.LEFT].reshape(2,-1)[1,:]==self.image_height//2].min()
        self.images_max_x[Pos.LEFT] = self.keeped_cylinder_points_2d[Pos.LEFT][0,self.keeped_cylinder_points_2d[Pos.LEFT].reshape(2,-1)[1,:]==self.image_height//2].max()

        # right camera
        calib_extrinsic_r_c_inv = linalg.tf_inv(self.extrinsic_calibrations[Pos.RIGHT])
        pt_in_cam_3 = self.new_matrices[Pos.RIGHT] @ (calib_extrinsic_r_c_inv @ self.cylinder_points)[:3,:]
        pt_in_cam_3d = (calib_extrinsic_r_c_inv @ self.cylinder_points)[:3,:]
        pt_in_cam = (cv2.projectPoints(pt_in_cam_3d,np.zeros((3, 1)),np.zeros((3, 1)), self.new_matrices[Pos.RIGHT], self.distortion_coefficients[Pos.RIGHT]*0.0))[0].reshape(-1,2).T
        keep = np.logical_and(np.logical_and(np.logical_and(np.logical_and(pt_in_cam[0,:] >=0, pt_in_cam[0,:] < image_center.shape[1]), pt_in_cam[1,:] >=0), pt_in_cam[1,:] < image_center.shape[0] ),pt_in_cam_3[2,:]>0)
        self.keeped_in_cam_points[Pos.RIGHT] = pt_in_cam[:,keep].astype(np.int) 
        self.keeped_cylinder_points_2d[Pos.RIGHT] = self.cylinder_points_2d[:,keep].astype(np.int)
        self.images_min_x[Pos.RIGHT] = self.keeped_cylinder_points_2d[Pos.RIGHT][0,self.keeped_cylinder_points_2d[Pos.RIGHT].reshape(2,-1)[1,:]==self.image_height//2].min()
        self.images_max_x[Pos.RIGHT] = self.keeped_cylinder_points_2d[Pos.RIGHT][0,self.keeped_cylinder_points_2d[Pos.RIGHT].reshape(2,-1)[1,:]==self.image_height//2].max()

    def __compute_mask(self):
        # generate fusion masks
        for pos in Pos:
            self.masks[pos] = np.zeros((self.image_height, self.image_width), dtype=np.float32)

        span_lc = (self.images_max_x[Pos.LEFT]-self.images_min_x[Pos.CENTER]) // 2
        center_lc = span_lc + self.images_min_x[Pos.CENTER]
        
        span_cr = (self.images_max_x[Pos.CENTER]-self.images_min_x[Pos.RIGHT]) // 2
        center_cr = span_cr + self.images_min_x[Pos.RIGHT]

        img_c_min_x2 = int(center_lc - span_lc * self.fusion_overlap_ratio)
        img_l_max_x2 = int(center_lc + span_lc * self.fusion_overlap_ratio)
        img_r_min_x2 = int(center_cr - span_cr * self.fusion_overlap_ratio)
        img_c_max_x2 = int(center_cr + span_cr * self.fusion_overlap_ratio)

        self.masks[Pos.LEFT][:,:img_c_min_x2] = 1.0
        self.masks[Pos.CENTER][:,img_l_max_x2:img_r_min_x2] = 1.0
        self.masks[Pos.RIGHT][:,img_c_max_x2:] = 1.0

        self.masks[Pos.LEFT][:,img_c_min_x2:img_l_max_x2]    = np.linspace(1.0, 0.0,img_l_max_x2-img_c_min_x2)
        self.masks[Pos.CENTER][:,img_c_min_x2:img_l_max_x2]  = np.linspace(0.0, 1.0,img_l_max_x2-img_c_min_x2)
        self.masks[Pos.CENTER][:,img_r_min_x2:img_c_max_x2]  = np.linspace(1.0, 0.0,img_c_max_x2-img_r_min_x2)
        self.masks[Pos.RIGHT][:,img_r_min_x2:img_c_max_x2]   = np.linspace(0.0, 1.0,img_c_max_x2-img_r_min_x2)


    def __get_cylinder_points(self, image_width, image_height, fov_width, fov_height, radius):
        # generate 2D and 3D cylinder masks
        angle_horizontal = np.linspace(-fov_width/2, fov_width/2,image_width)
        angle_vertical = np.linspace(-fov_height/2, fov_height/2,image_height)
        gh, gv = np.meshgrid(angle_horizontal, angle_vertical)
        gh = gh.ravel()
        gv = gv.ravel()

        px_x, px_y = np.arange(0,image_width),np.arange(0,image_height)
        gpx_x, gpx_y = np.meshgrid(px_x, px_y)
        gpx_x, gpx_y = gpx_x.ravel(), gpx_y.ravel()

        cylinder_points_2d = np.vstack([gpx_x, gpx_y])
        cylinder_points = np.vstack([np.sin(gh)*radius,radius * np.tan(gv),np.cos(gh)*radius, np.ones_like(gv)])

        return cylinder_points, cylinder_points_2d

    @staticmethod
    def gray_to_rgb(image):
        return np.stack([image,image,image], axis=2)

    def __are_keeped_in_cam_points_none(self):
        return self.keeped_in_cam_points[Pos.LEFT] is None and self.keeped_in_cam_points[Pos.CENTER] is None and self.keeped_in_cam_points[Pos.RIGHT] is None

    def __are_masks_none(self):
        return self.masks[Pos.LEFT] is None and self.masks[Pos.CENTER] is None and self.masks[Pos.RIGHT] is None

    def __assert_intrinsic_calibrations(self, intrinsic_calibrations):
        assert intrinsic_calibrations is not None, 'please specify camera intrinsic matrix left to right' 
        assert isinstance(intrinsic_calibrations, list), 'intrinsic need to be provide as a list left to right'
        assert len(intrinsic_calibrations)==3, 'need 3 intrinsic matrix left to right'

    def __assert_distortion_coefficients(self, distortion_coef):
        assert distortion_coef is not None, 'please specify camera distortion coefficients left to right' 
        assert isinstance(distortion_coef, list), 'distortion coefficients need to be provide as a list left to right'
        assert len(distortion_coef)==3, 'need 3 distortion coefficients left to right'
    
    def __assert_extrinsic_calibrations(self, extrinsic_calibrations):
        assert extrinsic_calibrations is not None, 'please specify camera extrinsic calibrations left to right' 
        assert isinstance(extrinsic_calibrations, list), 'extrinsic calibrations need to be provide as a list left to right'
        assert len(extrinsic_calibrations)==3, 'need 3 extrinsic calibrations left to right'

    def __assert_image(self, image):
        assert image is not None, 'please specify 3 camera image in a list from left to right' 
        assert isinstance(image, list), 'the 3 camera images need to be provide as a list from left to right'
        assert len(image)==3, 'need 3 camera img from left to right in a list'

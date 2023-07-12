import numpy as np
from scipy.spatial.transform import Rotation as R

from .telescope import Telescope
from .ctypes_wrapper import c_spherical_camera_coords
from .multi_threading_params import _NUM_THREADS

class Projection:
    "TBD - Cartesian coords to camera spherical coords"
    
             
    @staticmethod
    def solAngle_sphere(tel_pos, sphere_R, shower_coords):
        """
        Calculate solid angle of a sphere seen by voxel at a certain distance.
        """
        
        tel_dist = Telescope.calc_rel_dist(tel_pos, shower_coords)   
        
        return c_solAngle_sphere(tel_dist, sphere_R, _NUM_THREADS)
    
    
    @staticmethod
    def get_camera_cartesian(tel_az, tel_zenith, coords):
        """
        Convert cartesian shower coordinates into coordinate system
        of telescope. 
        """
        
        r1 = R.from_euler('Z', -1. * tel_az, degrees = True)
        r2 = R.from_euler('Y', 90. - tel_zenith, degrees = True)
        cartesian = np.linalg.multi_dot([r2.as_matrix(), r1.as_matrix(), coords.T]).T
        
        return cartesian
    
    
    @staticmethod
    def get_camera_spherical(tel_pos, tel_az, tel_zenith, coords):
        """"
        Convert cartesian coordinates into spherical coordinates
        in coordinate system of telescope.
        """
        
        cartesian = Projection.get_camera_cartesian(tel_az, tel_zenith, coords)
        tel_dist = Telescope.calc_rel_dist(tel_pos, coords)
    
        #return is theta, phi
        return c_spherical_camera_coords(cartesian, tel_dist, _NUM_THREADS) 
    
    
    @staticmethod
    def offset_to_angles(xoff, yoff, azimuth, altitude, focal_length):
        """
        ***Taken one to one from sim_telarray****
        
        Transform from the offset an object or image has in the
        camera plane of a telescope to the corresponding Az/Alt.
        """
        
        obj_azimuth = np.zeros_like(xoff)
        obj_altitude = np.zeros_like(yoff)

        d = np.sqrt(xoff*xoff+yoff*yoff)
        q = np.arctan(d/focal_length)

        sq = np.sin(q)
        xp1 = xoff * (sq/d)
        yp1 = yoff * (sq/d)
        zp1 = np.cos(q)

        cx = np.sin(altitude)
        sx = np.cos(altitude)

        xp0 = cx*xp1 - sx*zp1
        yp0 = yp1
        zp0 = sx*xp1 + cx*zp1

        obj_altitude = np.arcsin(zp0)
        obj_azimuth  = np.arctan2(yp0,-xp0) + azimuth

        mask_az1 = np.where(obj_azimuth < 0.)
        obj_azimuth[mask_az1] += 2.* np.pi
        mask_az2 = np.where(obj_azimuth >= 2.*np.pi)
        obj_azimuth[mask_az2] -= 2.*np.pi

        return np.rad2deg(obj_altitude) - np.rad2deg(altitude), np.rad2deg(obj_azimuth) - np.rad2deg(azimuth)

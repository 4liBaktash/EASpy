import numpy as np

from .auxiliary import Orthonormal_Vectors
from .ctypes_wrapper import c_voxel_coords
from .multi_threading_params import _NUM_THREADS

class Tracking:
    """
    Class to assign coordinates to voxels in a Cartesian coordinate system.
    Voxels are generated by dividing each cylindrical shell around the shower
    axis in equal parts.
    For each relative evolution stage and energy the midpoints of the voxels
    are given by the arithmetic mean of x[i] and x[i+1]. The shell width is
    then given by x[i+1] - x[i] such that the volume of the voxel at evolution
    stage t is:
    V = pi * (x[i+1]^2 - x[i]^2) * ds, where ds is the step size in slant distance
     

    Parameters
    ----------
    x0 : float
         x-coordinate of the impact position of the shower
         units: cm
    
    y0 : float
         y-coordinate of the impact position of the shower
         units: cm
    
    az:  flot
         azimuth angle of the shower direction
         units: degree
         
    
    Attributes:
    -----------
    coords : 4D numpy.ndarray
             contains the x,y and z coordinates of the midpoints
             of each voxel. 
             coords.shape = (len(t), len(x)-1, Nslice, 3), where
             Nslice describes the number of slices each cylindrical
             shell is equally divided by. 
             units: cm
               
    Methods:
    -------
    get_coords_shower_axis : get the coordiantes along the shower track
    
    get_rot_vectors : get the rotation vectors in order to distribute
                      the electrons uniformely in 2*pi around the shower
                      axis
    """
    
    def __init__(self, cfg):
        
        self.rbins_mid = (cfg.rbins_mid).copy()*1e2      #m->cm
        self.rho = (cfg.spherical_atmo_rho[cfg.prf_tmask]).copy()
        self.x_shower = (cfg.x_shower).copy()
        self.y_shower = (cfg.y_shower).copy()
        self.z_shower = (cfg.z_shower).copy()
        self.unit_shower = (cfg.unit_shower).copy()
        self.Nvoxel = cfg.Nvoxel
        
        
    @staticmethod               
    def get_coords_shower_axis(zenith, azimuth, x0, y0, _atmo_s, _prf_tmask):
        """
        Calculate coordinates along the shower track for tmin < t < tmax.
        """
        
        zd = np.deg2rad(zenith)
        az = np.deg2rad(azimuth)
        r = _atmo_s[-1] - _atmo_s[_prf_tmask]   #distance with origin at z0

        # z-axis always relative to telescope position/observation lvl
        z_shower = np.cos(zd) * r
        x_shower = x0 + np.sin(zd) * np.cos(az) * r
        y_shower = y0 + np.sin(zd) * np.sin(az) * r
        
        return x_shower, y_shower, z_shower
    
    
    @property
    def get_coords(self):
        """"
        Calculate coordinates of midpoints for voxels in cartesian
        coordinate system with origin=(0,0,0), where z = 0 means
        height of observation level
        Orientation of axis (from your perspective):
        positive x-axis: towards you
        positive y-axis: right 
        az : measured from x-axis towards y-axis counter-clockwise
        
        This means a shower with az = 0 deg and zenith = 90 deg
        comes from the positive x-axis and the shower axis is parallel
        to the x-axis.
        """

        V1, V2 = Orthonormal_Vectors(self.unit_shower)
        # Divide cylindrical shells into Nvoxel equally spaced parts
        phi = np.linspace(0., 2.*np.pi, self.Nvoxel, endpoint=False)
        out_cos = np.cos(phi)
        out_sin = np.sin(phi)

        # Calculate coordinates of midpoints for voxels
        shower  = np.array([self.x_shower, self.y_shower, self.z_shower])
        rot     = np.outer(V1, out_cos) + np.outer(V2, out_sin) 
    
        return c_voxel_coords(shower, rot, self.rbins_mid, _NUM_THREADS)

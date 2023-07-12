import numpy as np

from .ctypes_wrapper import c_norm
from .multi_threading_params import _NUM_THREADS

class Telescope:
    "TBD"
    
    def __init__(self, x, y, z, theta, az,
                 efficiency=None, apert=None, R_area=None, 
                 pix_FoV=None, int_time=None, Track = None):
    
        _telescope(self, x, y, z, theta, az, efficiency, apert, 
                   R_area, pix_FoV, int_time, Track)
        
    @staticmethod
    def calc_rel_dist(tel_pos, coords):
        #rel. distance voxels to telescope position 
        
        return c_norm(coords, tel_pos, _NUM_THREADS)
    
    
    @staticmethod
    def interp_qeff(wvl_interp, qeff, mirror_ref, camera_filter):
        """
        qeff : df for wvl dependent quantum efficiency
        wvl_interp : array of wvl's for interpolation
        """
        qeff_interp = np.interp(wvl_interp, qeff["wvl"].values, qeff["qe"].values, 
                                left = 0. , right = 0.)
        
        mirror_ref_interp = np.interp(wvl_interp, mirror_ref["wvl"].values, 
                                      mirror_ref["reflectivity"].values, 
                                      left = 0. , right = 0.)
        
        camera_filter_interp = np.interp(wvl_interp, camera_filter["wvl"].values, 
                                         camera_filter["eff"].values, 
                                         left = 0. , right = 0.)
        
        eff = qeff_interp * mirror_ref_interp * camera_filter_interp
        
        return eff
        
    
################################################################################
######################### Constructor ##########################################
################################################################################
def _telescope(telescope, x, y, z, theta, az, efficiency, apert, R_area,
               pix_FoV, int_time, Track):
    """
    Constructor of Telescope class and daughter classes.
    Parameters
    ----------
    telescope : Telescope
    x : float
        East coordinate of the telescope in cm.
    y : float
        North coordinate of the telescope in cm.
    z : float
        Height of the telescope in cm. 
    theta : float
        Zenith angle in degrees of the telescope pointing direction.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing direction.
    efficiency : DataFrame
        If given, the DataFrame should have two columns with wavelength in nm
        (with constant discretization step) and efficiency (decimal fraction).
    apert : float
        Angular diameter in degrees of the telescope field of view.
    area : float
        Radius of detection area (mirror area) in m 
    pix_FoV : float
        FoV of a single pixel in degree
    int_time : float
        Integration time in microseconds of camera frames.
    """        
    
    telescope.x = x 
    telescope.y = y 
    telescope.z = z 
    telescope.zenith = theta
    telescope.az = az
    telescope.R = R_area * 1e2   # m -> cm
    telescope.apert = apert
    telescope.Npix = np.rint(telescope.apert/pix_FoV).astype(int)
    telescope.qeff = efficiency
    
    return

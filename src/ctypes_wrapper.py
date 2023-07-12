import numpy as np
import ctypes
from ctypes import byref, c_int, c_int32, c_double, c_char_p

from .multi_threading_params import _NUM_THREADS, _NUM_THREADS_CHER

_libEASpy = ctypes.cdll.LoadLibrary('src/C_routines/libEASpy.so')

def c_pdf_x(zeta_0, zeta_1, x1, x, numthreads):
    """
    Calculates equation (14).
    zeta_0 : 2 dimensions (t, energy)
    zeta_1 : 1 dimension (t,)
    x1 : 1 dimension (energy,)
    x : 2 dimensions (t, number_of_steps_in_lateral)
    -------------------------------------------------
    numthreads : number of threads to use for OpenMP 
                 loop over t runs in parallel
    -------------------------------------------------
    out : 3 dimensions (t, energy, steps_in_lateral)
          pdf for lateral distribution
    -------------------------------------------------
    """

    dim_t, dim_E = zeta_0.shape 
    dim_x = x.shape[-1] 
    
    _libEASpy.pdf_x.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64, flags='aligned, c_contiguous',\
                                                      ndim = 2, shape = zeta_0.shape),\
                               np.ctypeslib.ndpointer(dtype = np.float64, flags='aligned, c_contiguous',\
                                                      ndim = 1, shape = zeta_1.shape),\
                               np.ctypeslib.ndpointer(dtype = np.float64, flags='aligned, c_contiguous',\
                                                      ndim = 1, shape = x1.shape),\
                               np.ctypeslib.ndpointer(dtype = np.float64, flags='aligned, c_contiguous',\
                                                      ndim = 2, shape = x.shape),\
                               ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                               np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                      ndim = 3, shape = (dim_t, dim_E,  dim_x)),\
                               ctypes.c_int]
    
   
    _libEASpy.pdf_x.restype  = ctypes.c_void_p
    
    out = np.empty((dim_t, dim_E, dim_x), dtype=np.float64)
    _libEASpy.pdf_x(zeta_0, zeta_1, x1, x, 
                    dim_t, dim_E, dim_x, out, numthreads)
    
    return out


def c_voxel_coords(shower, rot, r, numthreads):
    """
    Calculates voxel coordinates. 
    
    shower : coordinates along the shower axis for
             all relative evolution stages t
             
    rot : rotation vectors for describing voxel
          coordinates along phi direction 
          
    r : lateral distance from shower axis in cm
        (1 dimensional - const. for all t)
    ------------------------------------------------
    numthreads : number of threads to use for OpenMP 
                 loop over t runs in parallel
    ------------------------------------------------
    out : coordinates of all voxels 
          out.shape = (dim(t) * dim(rot) * dim(r), 3)
    -------------------------------------------------
    """
    
    dim_t = shower.shape[-1]
    dim_rot = rot.shape[-1]
    dim_r = r.shape[0]
    
    _libEASpy.voxel_coords.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64,\
                                                              ndim = 2, shape = shower.shape),\
                                       np.ctypeslib.ndpointer(dtype = np.float64,\
                                                              ndim = 2, shape = rot.shape),\
                                       np.ctypeslib.ndpointer(dtype = np.float64,\
                                                              ndim = 1, shape = r.shape),\
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int,\
                                       np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                              ndim = 2, shape = (dim_t*dim_rot*dim_r, 3)),\
                                       ctypes.c_int]
    
   
    _libEASpy.voxel_coords.restype  = ctypes.c_void_p
    
    out = np.empty((dim_t*dim_rot*dim_r, 3), dtype=np.float64)
    _libEASpy.voxel_coords(shower, rot, r, dim_t, 
                           dim_rot, dim_r, out, numthreads)
    
    return out


def c_norm(coords, tel_coords, numthreads):
    """
    Calculates distance between voxel coordinates and 
    telescope position.
    
    coords: coordinates of voxels
            (M, 3) dimensional
            
    tel_coords: coordinates of telescope
                (3,) dimensional
    ------------------------------------------------
    numthreads : number of threads to use for OpenMP 
                 loop over M runs in parallel
    ------------------------------------------------
    out: (M,) dimensional array containing the
          distances of voxel positions to telescope
    """
    
    dim  = coords.shape[0]
    
    _libEASpy.norm.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64,\
                                                      ndim = 2, shape = coords.shape),\
                               np.ctypeslib.ndpointer(dtype = np.float64,\
                                                      ndim = 1, shape = tel_coords.shape),\
                               ctypes.c_int,\
                               np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                      ndim = 1, shape = (dim)),\
                               ctypes.c_int]
    
   
    _libEASpy.norm.restype  = ctypes.c_void_p
    
    out = np.empty((dim), dtype=np.float64)
    _libEASpy.norm(coords, tel_coords, dim, out, numthreads)
    
    return out


def c_unit_center(coords, tel_coords, dist, numthreads):
    """
    Calculates distance between voxel coordinates and 
    telescope position.
    
    coords: coordinates of voxels
            (M, 3) dimensional
            
    tel_coords: coordinates of telescope
                (3,) dimensional
    ------------------------------------------------
    numthreads : number of threads to use for OpenMP 
                 loop over M runs in parallel
    ------------------------------------------------
    out: (M,) dimensional array containing the
          distances of voxel positions to telescope
    """
    
    dim  = coords.shape[0]
    
    _libEASpy.unit_center.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64,\
                                                             ndim = 2, shape = coords.shape),\
                                      np.ctypeslib.ndpointer(dtype = np.float64,\
                                                             ndim = 1, shape = tel_coords.shape),\
                                      np.ctypeslib.ndpointer(dtype = np.float64,\
                                                             ndim = 1, shape = dist.shape),\
                                      np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                             ndim = 2, shape = coords.shape),\
                                      ctypes.c_int, ctypes.c_int]
    
   
    _libEASpy.unit_center.restype  = ctypes.c_void_p
    
    out = np.empty_like(coords, dtype=np.float64)
    _libEASpy.unit_center(coords, tel_coords, dist, out, dim, numthreads)
    
    return out


def c_spherical_camera_coords(cartesian_camera, tel_dist, numthreads):
    """
    Calculates voxel positions in spherical coordinates
    in camera coordinate system. 
    
    cartesian_camera : cartesian coordinates of voxel positions
                       in camera coordinate system
                       (M,3) dimensional
                       
    tel_dist : distance of voxel positions to telescope
               (M,) dimensional
    ------------------------------------------------
    numthreads : number of threads to use for OpenMP 
                 loop over M runs in parallel
    ------------------------------------------------
    ******CARE******
    during the calculation the (i,3) dimension of
    cartesian_camera is modified! Check C-code for
    more info.
    Results for phi are stored in !!tel_dist!! to 
    save memory for large arrays.
    Make copy of both input arrays if you want to 
    keep using them!
    ****************
    out_phi : azimuthal coordinates of voxel positions
              in camera coordinate system
              (M,) dimensional
              
    out_theta : polar coordinates of voxel positions
                in camera coordinate system
                (M,) dimensional
    """

    dim  = tel_dist.shape[0]
    
    _libEASpy.spherical_camera_coords.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                          ndim = 1, shape = (dim)),\
                                                  np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                          ndim = 1, shape = (dim)),\
                                                  np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                          ndim = 1, shape = (dim)),\
                                                  np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                                          ndim = 1, shape = (dim)),\
                                                  ctypes.c_int,\
                                                  np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                                          ndim = 1, shape = (dim)),\
                                                  ctypes.c_int]
    
   
    _libEASpy.spherical_camera_coords.restype  = ctypes.c_void_p
    
    out_theta = np.empty((dim), dtype=np.float64)
    _libEASpy.spherical_camera_coords(cartesian_camera[:, 0], cartesian_camera[:, 1], 
                                      cartesian_camera[:, 2], tel_dist, dim, out_theta, 
                                      numthreads)
    
    return out_theta, tel_dist


def c_solAngle_sphere(tel_dist, R, numthreads):
    """"
    Calculates solid angle of sphere seen by a voxel at a certain distance.
    More information in C-code.
    
    tel_dist : tel_dist : distance of voxel positions to telescope
               (M,) dimensional
    
    R : radius of mirror
        double
    ------------------------------------------------
    numthreads : number of threads to use for OpenMP 
                 loop over M runs in parallel
    ------------------------------------------------
    out : solid angle of sphere seen by voxel 
          (M,) dimensional
    """

    dim  = tel_dist.shape[0]
    resu = np.empty((dim), dtype=np.float64)
    _libEASpy.c_solAngle_sphere.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                                   ndim = 1, shape = (dim)),\
                                            np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                   ndim = 1, shape = (dim)),\
                                            ctypes.c_double, ctypes.c_int, ctypes.c_int]
    
   
    _libEASpy.c_solAngle_sphere.restype  = ctypes.c_void_p
    
    _libEASpy.c_solAngle_sphere(resu, tel_dist, R, dim, numthreads)
    
    return resu


def c_arrival_times(axis_time, tel_dist):
    """
    computes arrival times in seconds.
    arrival_time = axis_time(=slant_distance/c) + dist_time(=tel_dist/c)
    
    axis_time : time lag along the shower axis [cm/s]
                (len(t),) dimensional
    
    tel_dist : distance from voxel positions to telescope [cm]
                (len(t) * len(x)-1 * Nvoxel,) dimensional
    ------------------------------------------------------------
    ****CARE****
    results are stored in !!tel_dist!!
    Make copy if you want to keep using it!
    ************
    out : arrival_times in seconds
          (len(t) * len(x)-1 * Nvoxel,) dimensional
    """
    
    dim_t  = axis_time.shape[0]
    dim_lat = int(tel_dist.shape[0]/dim_t)
    offsets = np.array([dim_lat*i for i in range(dim_t)], dtype = np.int32)
    
    _libEASpy.arrival_times.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64, 
                                                               ndim = 1, shape = (dim_t)),\
                                        np.ctypeslib.ndpointer(dtype = np.float64, flags='WRITEABLE',\
                                                               ndim = 1, shape = (dim_t * dim_lat)),\
                                        np.ctypeslib.ndpointer(dtype = np.int32,\
                                                               ndim = 1, shape = offsets.shape),\
                                        ctypes.c_int, ctypes.c_int]
    
   
    _libEASpy.arrival_times.restype  = ctypes.c_void_p
    
    _libEASpy.arrival_times(axis_time, tel_dist, offsets, dim_t, dim_lat)
    
    return tel_dist


def c_apply_qeff_absorbtion(LUT, wvl_idx, qeff, coords, Nph, cfg):
    """
    Applies quantum eff. and absorption to photons hitting the sphere. 
    
    LUT     : absorbtion look up table 
    wvl_idx : index of randomly drawn wavelengths
    qeff    : Table with wvl dependend quantum eff.
    coords  : coordinates of photons
    Nph     : number of photons
    """
    
    h0 = cfg.h0 
    impact = np.array([cfg.x0, cfg.y0, 0.])
    Telpos = cfg.tel_pos
    
    dim_t   = Nph.shape[0]
    dim_LUT = LUT.shape[-1]
    
    resu = np.empty(Nph.shape[0], dtype=np.float64)
    
    _libEASpy.apply_qeff_absorbtion.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                       ndim = 1, shape = resu.shape),\
                                                np.ctypeslib.ndpointer(dtype = np.float64, 
                                                               ndim = 2, shape = LUT.shape),\
                                                np.ctypeslib.ndpointer(dtype = np.int32,\
                                                               ndim = 1, shape = wvl_idx.shape),\
                                                np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                       ndim = 1, shape = qeff.shape),\
                                                np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                       ndim = 2, shape = coords.shape),\
                                                c_double,\
                                                np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                       ndim = 1, shape = impact.shape),\
                                                np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                       ndim = 1, shape = Nph.shape),\
                                                np.ctypeslib.ndpointer(dtype = np.float64,\
                                                                       ndim = 1, shape = Telpos.shape),\
                                                c_int, c_int, c_int]
                                        
    args = (resu,
            LUT,
            wvl_idx,
            qeff,
            coords,
            h0,
            impact,
            Nph,
            Telpos,
            dim_t, dim_LUT,
            _NUM_THREADS,
           )
   
    _libEASpy.apply_qeff_absorbtion.restype  = ctypes.c_void_p
    
    _libEASpy.apply_qeff_absorbtion(*args)
    
    return resu


class SI_params(ctypes.Structure):
    _fields_ = [
                  ('ageMinMax', ctypes.POINTER(c_int)),
                  ('unit_mom', ctypes.POINTER(c_double)),
                  ('tanAngle', ctypes.POINTER(c_double)),
                  ('Nch', ctypes.POINTER(c_double)),
                  ('r_theta_min', ctypes.POINTER(c_double)),
                  ('r_theta_max', ctypes.POINTER(c_double)),
                  ('d_sphere_mid', ctypes.POINTER(c_double)) 
               ]


class SI_DATA(ctypes.Structure):
    _fields_ = [  ('dim_t', c_int),
                  ('dim_r', c_int),
                  ('dim_rot', c_int),
                  ('N_Ebins', c_int),
                  ('R', c_double),
                  ('params', ctypes.POINTER(SI_params)),
               ]


    def __init__(self, data, dim_t, dim_r, dim_rot, R):
        NumberOfStructs = len(data.keys())
        self.N_Ebins = NumberOfStructs
        self.dim_t = dim_t
        self.dim_r = dim_r
        self.dim_rot = dim_rot
        self.R = R
        self.params = ctypes.cast((SI_params*NumberOfStructs)(), ctypes.POINTER(SI_params))

        for i in range(0, NumberOfStructs):
            dat = list(data.values())[i]
            self.params[i].ageMinMax  = np.ctypeslib.as_ctypes(dat[0])
            self.params[i].unit_mom   = np.ctypeslib.as_ctypes(dat[1].ravel())
            self.params[i].tanAngle   = np.ctypeslib.as_ctypes(dat[2])
            self.params[i].Nch        = np.ctypeslib.as_ctypes(dat[3].ravel()) 
            self.params[i].r_theta_min  = np.ctypeslib.as_ctypes(dat[4].ravel())
            self.params[i].r_theta_max  = np.ctypeslib.as_ctypes(dat[5].ravel())
            self.params[i].d_sphere_mid = np.ctypeslib.as_ctypes(dat[6].ravel())


def c_sphere_intersect(init_dict, dim_t, dim_r, dim_rot, R, unit_center, sphere_dist):
    """
    """
   
    resu = np.zeros_like(sphere_dist, dtype = np.float64)
    
    data_in = SI_DATA(init_dict, dim_t, dim_r, dim_rot, R)
    _libEASpy.cone_sphere_intersect.argtypes = [ctypes.POINTER(SI_DATA),
                                                ctypes.POINTER(c_double),
                                                ctypes.POINTER(c_double),
                                                ctypes.POINTER(c_double),
                                                c_int]


    _libEASpy.cone_sphere_intersect.restype = ctypes.c_void_p

    _libEASpy.cone_sphere_intersect(data_in, 
                                    np.ctypeslib.as_ctypes(unit_center),
                                    np.ctypeslib.as_ctypes(sphere_dist),
                                    np.ctypeslib.as_ctypes(resu), 
                                    _NUM_THREADS_CHER)
    return resu

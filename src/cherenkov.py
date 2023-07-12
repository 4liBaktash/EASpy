import numpy as np

from .auxiliary import calc_rel_dist, Point2Vector, Vec2Unit
from .ctypes_wrapper import c_unit_center, c_sphere_intersect
from .worker import Worker
from .multi_threading_params import _NUM_THREADS, _NUM_THREADS_CHER

class Cherenkov:
    
    def __init__(self, cfg):
        
        self.t = (cfg.prf_t).copy()
        self.Ebins_mid = (cfg.Ebins_mid).copy()
        self.rbins_mid = (cfg.rbins_mid).copy() * 1e2    #m->cm
        self.Nvoxel = cfg.Nvoxel
        self.rho = (cfg.spherical_atmo_rho[cfg.prf_tmask]).copy()
        self.delta = (cfg.spherical_atmo_delta[cfg.prf_tmask]).copy()
        self.ds = cfg.ds
        self.wvl_lo = cfg.wvl_lo
        self.wvl_up = cfg.wvl_up
        self.MomAngBins = (cfg.MomAngBins).copy()
        self.point_shower_axis = np.array([cfg.x_shower,
                                           cfg.y_shower, 
                                           cfg.z_shower])
        self.unit_shower = (cfg.unit_shower).copy()
        self.tel_pos = (cfg.tel_pos).copy()
        self.R = cfg.R * 1e2    #m->cm
        self.momentum_angle = self.calc_mean_theta
        self.cherenkov_angle, self.cherenkov_yield = self.get_yield_angle
        
    
    
    @property
    def pdf_theta(self):
        """
        Distribution of particle momentum angle
        Appendix A.2. eq. (8)
        theta : momentum angles in deg
        eps: energy in MeV
        returns: 
        !!n(t,eps,omega) = n(t, eps, theta)/sin(theta) !!
        """
        
        eps = self.Ebins_mid[:, np.newaxis]
        theta = self.MomAngBins[np.newaxis, :]

        b1 = -3.73 + 0.92*eps**0.210
        b2 = 32.9 - 4.84*np.log(eps)
        alpha1 = -0.399
        alpha2 = -8.36 + 0.44*np.log(eps)
        sigma = 3.

        tmp_norm_arr  = (np.exp(b1) * theta**alpha1)**(-1./sigma)
        tmp_norm_arr += (np.exp(b2) * theta**alpha2)**(-1./sigma)

        norm_arr  = (tmp_norm_arr)**(-1. * sigma)
        norm_arr /= np.trapz(norm_arr, theta, axis = -1)[:, np.newaxis]

        return norm_arr

    
    @property
    def calc_mean_theta(self):
        """
        Returns expectation value for momentum angle in radians for each energy
        """
        
        pdf = self.pdf_theta
        resu = np.trapz(pdf * self.MomAngBins, self.MomAngBins, axis = 1)  

        return np.deg2rad(resu)
    
    
    def calc_yield_angle(self, age, energy, delta, rho):
        """"
        Taken from CORSIKA.
        Code below assumes that electrons are moving parallel to shower axis.
        Correction which also takes into account the angle to the shower axis,
        i.e. 1/cos(mom_angle), is done later in function init_sphere_intersection.
        """
        
        me = 0.511
        finestructure = 1.37035999E+02
        NM2CM = 1e7
        dX = self.ds * self.rho

        CYIELD  = (self.wvl_up - self.wvl_lo)
        CYIELD /= self.wvl_up * self.wvl_lo 
        CYIELD *= 2.*np.pi/finestructure * NM2CM
        
        BETA = np.sqrt( (1.-(me/(energy+me))) * (1.+(me/(energy+me))) )

        for i in age:
            ETA1  = 1. + delta[i]
            BEMX   = BETA*ETA1

            PHOTCT = CYIELD * self.ds * (1. - 1./BEMX**2)
            mask = np.where(BETA*ETA1 > 1.)
            cher_angle = np.zeros(len(BEMX))
            cher_angle[mask] = np.arccos(1./(BEMX[mask]))

            yield PHOTCT, mask, cher_angle

            
    @property    
    def get_yield_angle(self):
        """
        dN_gamma/dX = integral_log(E_thr) (N*n * y(h,E) *dlog(E))
        """
        
        return_cher_angle = np.zeros((len(self.t), len(self.Ebins_mid)))
        return_cher_yield = np.zeros((len(self.t), len(self.Ebins_mid)))
        _range_age = range(len(self.t))
        
        cher_yield_angle = self.calc_yield_angle(list(_range_age), self.Ebins_mid,\
                                                 self.delta, self.rho)

        for i in _range_age:

            tmp_yield, tmp_mask, tmp_cher_angle = next(cher_yield_angle)

            return_cher_angle[i][tmp_mask] = tmp_cher_angle[tmp_mask]
            return_cher_yield[i][tmp_mask] = tmp_yield[tmp_mask]

        return return_cher_angle, return_cher_yield
    
    
    def calc_mom_vec(self, mom_angle, cone_origin = None):
        """"
                    p1 |
                       |\
                    s  |o\  o = mom_angle
                       |  \
         point_shower  |___\ cone_origin
                       | r
                       |
        
        mom_angle: float, momentum angle of particle in rad
        cone_origin: numpy.ndarray, (300,3) coordinates of origin of cherenkov cone
        
        Since return is a unit vector, cone_origin can be at any arbitrary relative shower
        age and lateral bin.
        """
        s = self.rbins_mid[0] * 1./np.tan(mom_angle)
        p1 = self.point_shower_axis.T[0] + s*self.unit_shower
        
        if(hasattr(self, "cone_origin")):
            mom_vec = Point2Vector(p1[np.newaxis, :], self.cone_origin)
        else:
            if(cone_origin is not None):
                mom_vec = Point2Vector(p1[np.newaxis, :], cone_origin)
            else:
                raise ValueError("cone_origin not defined!")
        
        unit_mom_vec = Vec2Unit(mom_vec)
        
        return unit_mom_vec
    
    
    def calc_sphere_minmax(self, tanAngle):
        """
        Calculate radius [cm] of inner/outer cherenkov circle in showerplane such that the 
        midpoint of the sphere is contained within this plane. Also calculate the
        distance [cm] to sphere midpoint.
        """
        s    = self.rbins_mid[np.newaxis, ...] * 1./tanAngle[..., np.newaxis]
        p1   = self.point_shower_axis.T[:, np.newaxis, :] + s[..., np.newaxis]*self.unit_shower
        CmP1 = self.tel_pos - p1
        
        length_CmP1 = np.linalg.norm(CmP1, axis = -1)
        unit_CmP1   = CmP1/length_CmP1[..., np.newaxis]
        # ... *-1.* ... : unit_shower is pointing upwards
        l = np.sum(self.unit_shower*-1.*unit_CmP1, axis = -1) * length_CmP1
        
        d_sphere_mid  = np.sqrt(length_CmP1**2 - l**2)
        r_thetaMinMax = tanAngle[..., np.newaxis] * l
        
        return d_sphere_mid, r_thetaMinMax
    
    
    def init_sphere_intersection(self, Ebin):
        """"
        Calculate energy dependent parameters for interesection code.
        Ebin : int, index of energy bin 
        """
        
        #particle momentum angle and cherenkov angle for fixed energy
        mom_angle = self.momentum_angle[Ebin]
        cher_angle = self.cherenkov_angle[:, Ebin]
        
        #maximum and minimum angle of cherenkov photons with shower axis
        theta_min = mom_angle - cher_angle
        theta_max = mom_angle + cher_angle
        
        #distance of circle midpoints and radius of inner/outer cherenkov circle
        #(cherenkov ring is formed by inner and outer circle)
        d_sphere_mid, r_theta_min = self.calc_sphere_minmax(np.tan(theta_min))
        d_sphere_mid, r_theta_max = self.calc_sphere_minmax(np.tan(theta_max))
        
        #at fixed energy the particle might be over threshold at later stages
        ageMinMax = np.zeros(2, dtype = np.int32)
        ageMinMax[0] = np.nonzero(cher_angle)[0][0]
        ageMinMax[1] = np.nonzero(cher_angle)[0][-1] + 1

        #number of produced cherenkov photons at fixed energy 
        #w.r.t shower age and lateral distance
        #photons are distributed equally over number of voxels
        Nch  = self._Nch[:, Ebin, :].copy()/self.Nvoxel
        #path of electron has angle to shower axis
        Nch /= np.cos(mom_angle)
        #calculate unit momentum vector 
        unit_mom_vec = self.calc_mom_vec(mom_angle)

        return ageMinMax, unit_mom_vec, np.tan(cher_angle), Nch,\
               r_theta_min/1e5, r_theta_max/1e5, d_sphere_mid/1e5
    
    
    def calc_Nph(self, coords, xSamples):
        """"
        Calculate number of cherenkov photons hitting the sphere
        coords = np.ndarray, coordinates of voxel positions [cm]
        xSamples = np.ndarray, distribution of particles w.r.t shower age,
                               lateral distance and energy
        """
        
        self.calc_Nph_shower(xSamples)
      
        _shape = len(self.t), len(self.rbins_mid), self.Nvoxel, 3
        coords_reshaped = coords.reshape(_shape)
        self.cone_origin = coords_reshaped[0,0,:,:]
        sphere_dist = calc_rel_dist(self.tel_pos, coords)
        unit_center = c_unit_center(coords, self.tel_pos, sphere_dist, _NUM_THREADS)
        
        mask_Thresh = np.unique(np.where(self.cherenkov_angle > 0.)[1])
        #initialize sphere intersection data
        w = Worker(_NUM_THREADS_CHER)
        init_dict = w.run(self.init_sphere_intersection, mask_Thresh) 

        #run intersection code
        args = (init_dict, 
                len(self.t), 
                len(self.rbins_mid), 
                self.Nvoxel, 
                self.R/1e5,
                unit_center.ravel(),
                sphere_dist/1e5)
        
        print(self)
        return c_sphere_intersect(*args)
    
    
    def calc_Nph_shower(self, xSamples):
        """
        Calculate number of cherenkov photons produced w.r.t shower age, particle
        energy and lateral distance
        """
        self._Nch  = (xSamples["electron"] + xSamples["positron"]) 
        self._Nch *= self.cherenkov_yield[..., np.newaxis]
        
        return
    
    
    @staticmethod
    def draw_random_wvl(wvl_lo, wvl_up, size, wvl_bin_lo):
        """
        Draw random wvl according to 1/lambda**2 distribution
        
        wvl_lo: int, lower wavelength limit [nm]
        wvl_up: int, upper wavelength limit [nm]
        size: int, determines how many random numbers to draw
        
        return: np.ndarray, index of random wavelengths assuming that 
                wvl_bins = [wvl_lo, wvl_lo + 1, ....., wvl_up - 1, wvl_up]
        """
        
        #Taken from simtel_array
        w = Worker(mode = "ThreadRNG")
        random_number = w.run_RNG_random(size)
        #According to 1./lambda^2 distribution
        wvl = (1./(1./wvl_lo - random_number*(1./wvl_lo - 1./wvl_up)))

        return wvl.astype(int) - wvl_bin_lo
    
    
    def __str__(self):
        Xmax_index = np.where(self.t == 0.)[0][0]
        E = self.Ebins_mid[0::10]
        momAng = np.rad2deg(self.momentum_angle[0::10])
        cherAng = np.rad2deg(self.cherenkov_angle[Xmax_index][0::10])
        nph = np.sum(self._Nch[Xmax_index][0::10], axis = -1)
        _resu = ""
        
        for e,m,c,n in zip(E, momAng, cherAng, nph):
            _resu += "%.2e              %.2e                 %.2e              %.2e\n" % (e, m, c,n)
            
        resu = "STARTING WITH CHERENKOV CHAIN\n"\
        "Momentum angle of particles vs. cherenkov angle at shower Maximum\n"\
        "Energy [MeV]     Momentum angle [deg]    Cherenkov angle [deg]   Cherenkov photons\n"
        
        return resu + _resu

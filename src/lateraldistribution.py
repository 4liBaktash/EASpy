import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtpz

from .ctypes_wrapper import c_pdf_x
from .multi_threading_params import _NUM_THREADS
from .worker import Worker

class LateralDistribution:
    """ 
    Calculates lateral distribution from shower axis w.r.t to log(x), 
    where x is given by x = r/r_M with r_M Moliére radius.
    See A.4 in Appendix.
    Parameters given by equation (A.3)
    Lateral distribution depends on realtive evolution stage and energy :
    shower age vs. energy vs. lateral distance
    
    
    Parameters
    ----------
    r_steps : int
        stepsize for lateral distance in meter
        minimum r = 1m
        maximum r = 1km
        
        
    Attributes:
    ----------
    pdf : Dictionary with keys = electron, positron
          Each key contains a 3D numpy.ndarray which describes the
          normalized pdf w.r.t log(x) for each energy provided by 
          the EnergySpectrum object. 
          n(t; log(E), log(x)) = 1./N(t;log(E)) * d^2N(t)/(dlog(E)dlog(E)), with
          N(t; log(E)) = dN(t)/dlog(E)
          pdf.shape = (len(t), len(E), len(x))

    xSamples : Dictionary with keys = electron, positron
               Each key contains a 3D numpy.ndarray which describes
               the number of expected particles at each evolution stage
               t, each energy E and each lateral distance x, i.e.
               xSamples[0,0,0] describes the number of particles at
               t=tmin, E=Emin, x=xmin.
               xSamples.shape = (len(t), len(E), len(x))
               
               
    Methods:
    -------
    calc_pdf_x: Returns the calculated pdf 
    calc_x: Calculate x values for constant stepsize in r
    """
    
    def __init__(self, cfg):
        
        self.eps = (cfg.Ebins).copy()
        self.t = (cfg.prf_t).copy()
        self.x = (cfg.x).copy()
        self._rbinsMid = (cfg.rbins_mid).copy() 
        self._EbinsMid = (cfg.Ebins_mid).copy()
        
    @property
    def calc_pdf_x(self):
        """
        See Appendix A.4, equation (14) and A.3
        pdf is normalized for each energy bin w.r.t log(x)
        pdf.shape = (len(t), len(energy), len(x))
        """
        
        x1      = 0.859 - 0.0461*np.log(self.eps)**2 + 0.00428 * np.log(self.eps)**3
        zeta_t  = 0.0263*self.t
        zeta_0  = zeta_t[:, np.newaxis] + 1.34 + 0.160 * np.log(self.eps)[np.newaxis,:] 
        zeta_0 -= 0.0404 * (np.log(self.eps)**2)[np.newaxis, :]
        zeta_0 += 0.00276 * (np.log(self.eps)**3)[np.newaxis, :]
        zeta_1  = zeta_t - 4.33
        
        norm_arr  = c_pdf_x(zeta_0, zeta_1, x1, self.x, _NUM_THREADS)
        norm_arr /= (np.trapz(norm_arr, np.log(self.x)[:, np.newaxis, :], axis = 2))[..., np.newaxis]

        return norm_arr
    
    
    @staticmethod
    def calc_x(_atmo_rho, _prf_tmask, rbins):
        """
        Calculate x values such that we have constant steps in physical units
        over the whole relative evolution of the shower
        x.shape = (len(t), int((1e3-1)/r_step))
        """
        
        rho = _atmo_rho[_prf_tmask]
        r_M = (9.6/rho)
        r = rbins * 1e2    #m->cm
        x = r/r_M[:, np.newaxis]
        
        return x 
        
    
    def integrate_pdf(self, pdf, N_logE, key):
        """
        Integrate pdf over energy and x.
        key = str, electron or positron.
        """
        # first integrate over log(energy) for each x 
        N_logx = np.diff(cumtpz(np.transpose(pdf * N_logE[key][:,:,np.newaxis], axes=(0,2,1)), 
                                  np.log(self.eps), axis = 2, initial = 0.), 
                         axis = 2)
        # now integrate over log(x) for each energy
        resu = np.diff(cumtpz(np.transpose(N_logx, axes=(0,2,1)), 
                              np.log(self.x[:, np.newaxis, :]), axis = 2, initial = 0.), 
                         axis = 2)
        return resu
    
    
    def get_xSamples(self, N_logE):
        """ 
        Calculates the number of expected particles at each relative evolution stage t,
        energy E and lateral distance from the shower axis x.
        This is done by first integrating the pdf to get the cdf and then obtaining the
        probabilities from the cdf.
        xSamples.shape = (len(t), len(energy)-1, len(x)-1)
        """
        
        pdf = self.calc_pdf_x
        w = Worker()
        selector = ["electron", "positron"]
        resu = w.run_with_wait(self.integrate_pdf, selector, pdf, N_logE)
        
        self.log_output(resu) 
        return resu
    
    
    @staticmethod
    def xSamples_transposed(xSamples):
        """
        xSamples.shape = (len(t), len(energy)-1, len(x)-1)
        return.shape = len(t), len(x)-1, len(energy)-1)
        """
        
        resu = {}
        for key in xSamples.keys():
            resu[key] = np.transpose(xSamples[key], axes=(0, 2, 1)) 
               
        return resu
    
    
    @staticmethod
    def get_NperE(xSamples):
        """
        Calculate number of particles w.r.t energy and shower age
        by summing up the contribution over lateral distance bins.
        """
       
        resu = {}
        for key in xSamples.keys():
            resu[key] = np.sum(xSamples[key], axis = 2)
         
        return resu
    
    
    @staticmethod
    def calc_pdf2D_x(age, x):
        """
        See 3D version - this time only shower stage vs. lateral distance.
        eq. (13)
        """

        zeta_0 = 0.0238 * age + 1.069
        zeta_1 = 0.0238 * age - 2.918
        x1     = 0.430
        
        norm_arr = x**zeta_0[:, np.newaxis] * (x1 + x)**zeta_1[:, np.newaxis]
        norm_arr /= (np.trapz(norm_arr, np.log(x), axis = 1))[:, np.newaxis]
    
        return norm_arr
    
    
    def log_output(self, resu):
        Xmax_index = np.where(self.t == 0.)[0][0]
        E = self._EbinsMid[0::10]
        r_index = int((len(self.x[0]) - 1)/5)
        
        print("Energy distribution of particles at shower maximum:")
        for key in ["electron", "positron"]:
            N = resu[key][Xmax_index][0::10]
            print(f"{key.upper()}:")
            print("Energy[MeV] %.1em    %.1em    %.1em    %.1em    %.1em" % tuple(self._rbinsMid[0::r_index]))
            for e,n in zip(E,N):
                print("%.2e    %.2e    %.2e    %.2e    %.2e    %.2e" % (e, *list(np.cumsum(n)[0::r_index])))
            print("\n")
        
        return

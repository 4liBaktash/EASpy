import numpy as np


class EnergySpectrum:
    """"
    Class to calculate the energy distribution with the parametrization given in:
    https://arxiv.org/pdf/0902.0548.pdf
    Energy spectrum for electrons and positrons is calculated depending on relative evolution stage t, 
    see eq. (6).
    
    Parameters:
    ----------
    energy : numpy.ndarray
        Desired energy range in MeV
     
     
    Attributes:
    ----------
     N_logE : Dictionary with following keys:
     
     -electron: Total differential number of electrons w.r.t
                dlog(energy).
                dN/d(logE) = N(t) * n(t; log(E))
     
     -positron: Total differential number of electrons w.r.t
                dlog(energy).
                dN/d(logE) = N(t) * n(t; log(E))
                     
     pdf : Dictionary with following keys:
     
     -electron: pdf for electrons, distribution is normalized w.r.t
                dlog(energy). 
                Describes the normalized differential number of particles 
                w.r.t dlog(energy) for each relative evolution stage : 
                n(t;log(E)) = 1/N(t) * dN/d(logE)
                
     -positron: pdf for positrons, distribution is normalized w.r.t
                dlog(energy).
                Describes the normalized differential number of particles 
                w.r.t dlog(energy) for each relative evolution stage : 
                n(t;log(E)) = 1/N(t) * dN/d(logE)
               
     profile : Profile object that is generated
     
     
     Methods:
     -------
     calc_pdf_Edist : calculate the probability density function (^= n(t, log(E)))
     """
    
    
    def __init__(self, cfg):
                
        self.energy = (cfg.Ebins).copy()
        self.t = (cfg.prf_t).copy()
        self.N = (cfg.prf_N).copy()
        
    
    @property
    def calc_pdf_Edist(self):
        """
        Calculates pdf by integrating non-normalized pdf w.r.t to log(energy),
        where energy is in units of MeV. Ecut is therefore given by energy[0].
        Normalization for electrons and positrons is given by "total" but pdf
        for "total" is not saved to dictionary.
        """

        # dictionary with keys = particles containing the pdf for each particle
        return_dict = {}
        # start with "total" to get the correct normalization
        particle_type = ["total", "electron", "positron"]

        for i in range(len(particle_type)):
            particle = particle_type[i]
            #see Appendix A.1 Table A.1
            if(particle == "electron"):
                A0     = 0.485 * np.exp(0.183*self.t - 8.17*self.t**2*1e-4)
                eps1   = 3.22 - 0.0068 * self.t
                eps2   = 106. - self.t
                gamma1 = 1.
                gamma2 = 1. + 0.0372 * self.t
                A0    *= np.array(norm)

            if(particle == "positron"):
                A0     = 0.516 * np.exp(0.201*self.t - 5.42*self.t**2*1e-4)
                eps1   = 4.36 - 0.0663 * self.t
                eps2   = 143. - 0.15 * self.t
                gamma1 = 2.
                gamma2 = 1. + 0.0374 * self.t
                A0    *= np.array(norm)

            if(particle == "total"):
                A0     = np.exp(0.191*self.t - 6.91*self.t**2*1e-4)
                eps1   = 5.64 - 0.0663 * self.t
                eps2   = 123. - 0.70 * self.t
                gamma1 = 1.
                gamma2 = 1. + 0.0374 * self.t
                norm   = list(map(getA1, np.tile(self.energy, (len(self.t),1)), A0, 
                                  eps1, eps2, np.full(len(self.t),gamma1), gamma2))
                A0    *= np.array(norm)

            if(particle != "total"):   
                n = np.apply_along_axis(getA1, 1, self.energy[:, np.newaxis],
                                        A0, eps1, eps2, gamma1, gamma2, False)

                return_dict[f"{particle}"] = n.T
                
        return return_dict
    
    
    @property
    def get_Edist(self):
        """"
        Calculates dN/d(logE) = N(t) * n(t; log(E))
        """
        
        pdf = self.calc_pdf_Edist
        N_logE = {}

        for key in pdf.keys():
            N_logE[key] = pdf[key] * self.N[:, np.newaxis]

        return N_logE


    
################################################################################
######################### Subroutines ##########################################
################################################################################
def getA1(eps, A0, eps1, eps2, gamma1, gamma2, normalize=True):
    """
    n = n(t, log(eps)) , parametrization given by Eq. (6)
    Return: n(t, log(eps)) = 1./N(t) * dN(t)/dlog(energy)
    """
    
    n  = A0 * eps**gamma1
    n /= (eps + eps1)**gamma1 * (eps + eps2)**gamma2
    
    if(normalize==True):
        resu = 1./np.trapz(n, np.log(eps))
    else:
        resu = n
        
    return resu

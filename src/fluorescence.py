import numpy as np
from matplotlib import pyplot as plt


class Fluorescence:
    """"
    Calculate number of photons due to Fluorescence by 
    taking the Fluorescence Yields provided by the
    AIRFLY experiment and deposit energy per relative
    evolution stage.
    
    Attributes:
    ----------
    normed : 1D numpy.ndarray
             Contains the normalized model. The contribution
             from each wavelength is summed up for each relative
             evolution stage t.
             normed.shape = (len(t),)
             
    N_perE : Dictionary with keys = electron, positron
             Each key contains a 2D numpy.ndarray which describes
             the contribution for each energy to the total number 
             of produced Fluorescence photons.
             N_perE.shape = (len(t), len(E))
    
    N_tot : 1D numpy.ndarray
            Contains the total number of Fluoresence photons for
            each relative evolution stage.
            
    Methods:
    -------
    show : Show the production of fluorescence photons as a function of slant depth.
    """
    # Parameters of the fluorescence model (34 bands)
    #       wvl(nm), Irel,  PP0,  PPw,   a
    model = ((296, 0.0516, 18.50, 0.00,  0.00),
            (298, 0.0277, 17.30, 0.00,  0.00),
            (302, 0.0041, 21.00, 0.00,  0.00),
            (308, 0.0144, 21.00, 0.00,  0.00),
            (312, 0.0724, 18.70, 0.00,  0.00),
            (314, 0.1105, 12.27, 1.20, -0.13),
            (316, 0.3933, 11.88, 1.10, -0.19),
            (318, 0.0046, 21.00, 0.00,  0.00),
            (327, 0.0080, 19.00, 0.00,  0.00),
            (329, 0.0380, 20.70, 0.00,  0.00),
            (331, 0.0215, 16.90, 0.00,  0.00),
            (334, 0.0402, 15.50, 0.00,  0.00),
            (337, 1.0000, 15.89, 1.28, -0.35),
            (346, 0.0174, 21.00, 0.00,  0.00),
            (350, 0.0279, 15.20, 1.50, -0.38),
            (354, 0.2135, 12.70, 1.27, -0.22),
            (358, 0.6741, 15.39, 1.30, -0.35),
            (366, 0.0113, 21.00, 0.00,  0.00),
            (367, 0.0054, 19.00, 0.00,  0.00),
            (371, 0.0497, 14.80, 1.30, -0.24),
            (376, 0.1787, 12.82, 1.10, -0.17),
            (381, 0.2720, 16.51, 1.40, -0.34),
            (386, 0.0050, 19.00, 0.00,  0.00),
            (388, 0.0117,  7.60, 0.00,  0.00),
            (389, 0.0083,  3.90, 0.00,  0.00),
            (391, 0.2800,  2.94, 0.33, -0.79),
            (394, 0.0336, 13.70, 1.20, -0.20),
            (400, 0.0838, 13.60, 1.10, -0.20),
            (405, 0.0807, 17.80, 1.50, -0.37),
            (414, 0.0049, 19.00, 0.00,  0.00),
            (420, 0.0175, 13.80, 0.00,  0.00),
            (424, 0.0104,  3.90, 0.00,  0.00),
            (427, 0.0708,  6.38, 0.00,  0.00),
            (428, 0.0494,  2.89, 0.60, -0.54))
    
    
    # Reference atmospheric conditions
    P0 = 800.  # mbar
    T0 = 293.  # Kelvin
    Y0_337 = 7.04 # yield of 337nm in dry air @ P0,T0
    
    def __init__(self, cfg, xSamples):
        
        Edep = EnergyDeposit(cfg, xSamples)
        
        self.N_tot = None
        self.Temp = (cfg.spherical_atmo_T[cfg.prf_tmask]).copy()
        self.P = (cfg.spherical_atmo_P[cfg.prf_tmask]).copy()
        self.rho = (cfg.spherical_atmo_rho[cfg.prf_tmask]).copy()
        self.X = (cfg.prf_X).copy()
        self.t = (cfg.prf_t).copy()
        self.ds = cfg.ds
        self.Nvoxel = cfg.Nvoxel
        self.tel_pos = (cfg.tel_pos).copy()
        self.R = cfg.R*1e2    #m->cm
        self.dEdX_perE = Edep.dEdX_perE.copy()
        if(cfg.Ecut):
            arrlen = len(cfg.Edep_Ecut)
            self.Edep_EcutInterp = np.interp(self.X, cfg.X_in_Ecut[:arrlen], cfg.Edep_Ecut)
            self.x = (cfg.x).copy()
    
    
    @property    
    def show(self):
        """
        Show the production of fluorescence photons as a function of slant depth.
        Returns
        -------
        ax : AxesSubplot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        ax.plot(self.X, self.N_tot, color = "black", 
                ls = "solid", label = "total")
        ax.axes.set_yscale("log")
        ax.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax.axes.yaxis.set_label_text("N_ph")
        ax.axes.legend()
        
        plt.show()
        return  ax
    
    
    @classmethod
    def get_normalized_model(cls, Temp, P):
    
        # We can draw the wavelength of the photons randomly later since we have Irel 
        # values.
        # For now treat the photons wavelength independent
        model = np.array(cls.model).ravel()

        nom  = model[1::5]
        nom *= (1. + cls.P0/model[2::5])
        denom  = P[..., np.newaxis]/model[2::5]
        denom *= (cls.T0/Temp[..., np.newaxis])**(0.5 - model[4::5])
        
        resu = nom/(1.+denom)
        
        return np.sum(resu, axis = 1)
    
    
    @property  
    def Nph_perE(self):
        """
        We want each relative evolution stage t to be the midpoint of emission
        This way everything is consistent since we calculated rho, T, P and
        and the model at this specific slant depth. This means the particles need
        to move half ds before t and half ds after t, while transversing the mass density
        rho which we calculated at t. The dX which we calculated in the Atmosphere class
        is the slant depth corresponding to each spherical shell - and not between the midpoints
        of the shells.
        (the error if you would take Atmo.dX is tiny if you chose sufficiently small steps in ds)    
        """
        
        Fl_NperE = {}
        
        dX = self.ds * self.rho
        normed = Fluorescence.get_normalized_model(self.Temp, self.P)

        for key in self.dEdX_perE.keys():
            NperE = self.Y0_337
            # N_ph/MeV * MeV/dX = N_ph/dX
            NperE *= self.dEdX_perE[key] * normed[:, np.newaxis]
            # N_photons * dX = N_ph
            NperE *= dX[:, np.newaxis]
            # Number of photons per energy and per particle type
            Fl_NperE[key] = NperE

        # Total number of photons per slant depth
        self.N_tot = np.sum(Fl_NperE["electron"], axis = 1) + np.sum(Fl_NperE["positron"], axis = 1)
        max_number = np.sum(self.N_tot)
        print(f"Total number of produced fluorescence photons: {max_number:.2e}")
        print(f"Maximum of fluorescence photons reached at shower stage: {self.t[np.argmax(self.N_tot)]:.2f}")

        return Fl_NperE
    
    
    @staticmethod
    def draw_random_wvl(size, wvl_interp):
        
        #get wvl bands and probabilities
        wvl = np.array(Fluorescence.model)[:,0].astype(int)
        prob = np.array(Fluorescence.model)[:,1]
        prob /= np.sum(prob)
        
        #draw appropiate indices of wvl_interp instead of real wvl's
        pp = np.zeros_like(wvl_interp, dtype = float)
        idx = np.searchsorted(wvl_interp, wvl)
        np.put(pp, ind = idx, v = prob)
        w = Worker(mode = "ThreadRNG")
        #wvl_idx = np.random.default_rng().choice(range(wvl_interp.shape[0]), size = size, p = pp)
        wvl_idx = w.run_RNG_choice(range(wvl_interp.shape[0]), size, pp)
        
        return wvl_idx
    
   
    def Nph_per_shell(self, xSamples):
        """
        Calculate number of fluorescence photons in each shell.
        """    
        
        Fl_NperE = self.Nph_perE
        NperE = LateralDistribution.get_NperE(xSamples)
        xSamples_transposed = LateralDistribution.xSamples_transposed(xSamples)
        
        for key in Fl_NperE.keys():

            Nph_per_Nparticle = 0.
            Nph_perE_shell = 0.
            Nph_shell = 0.

            # First calculate the number of photons emitted per particle per Energy
            Nph_per_Nparticle = Fl_NperE[key]/NperE[key]
            # Now calculate how many photons we have in each shell at distance x
            Nph_perE_shell = xSamples_transposed[key] * Nph_per_Nparticle[:, np.newaxis, :]
            # Sum up the contribution from the individual energies 
            Nph_shell = np.sum(Nph_perE_shell, axis = 2)  

            if(key == "electron"):
                resu = np.zeros_like(Nph_shell)
            resu += Nph_shell
        
        return resu
    
    
    def calc_Nph(self, coords, xSamples):
        """
        Calculate number of fluorescence photons hitting the sphere.
        """
        self.Nph_shell = self.Nph_per_shell(xSamples)
        if(hasattr(self, "Edep_EcutInterp")):
            self.Nph_shell += Ecut.Nph_per_shell(self.Edep_EcutInterp, self.t, self.x,\
                                                 self.Temp, self.P, self.ds, self.rho)
        
        resu  = Projection.solAngle_sphere(self.tel_pos, self.R, coords)
        #In case of fluorescence we have 4pi emission
        resu /= 4.*np.pi  
        resu *= np.repeat((self.Nph_shell/self.Nvoxel).ravel(), self.Nvoxel)
        
        return resu

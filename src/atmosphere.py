import numpy as np

class Atmosphere:
    """"
    Atmosphere class which will generate a discretization with the desired
    slant depth steps.
    
    
    Parameters:
    -----------
    df : Pandas.Dataframe
         Should contain values for desired atmospheric model 
         columns: T [K], rho [g/cm**3], P [mbar], P/P_w, h [km]
         for vertical(!) slant depth
          
    h0 : float
        height of observation in units of km
    
    zenith : float
        zenith angle of shower in degree
        
    ds : float
        distance the ray takes in each spherical shell 
        of the discretized atmosphere
        units : cm
        
    steps_h : int
        determines the width of the spherical shells
        each shell will have a width of (htop-h0)/steps_h
      
      
    Attributes:
    -----------
    atmosphere: Dataframe with following columns:
    
    h : float
        height in cm above sea level
    
    X : float
        slant depth in units of g/cm**2
    
    rho : float
        density in units of g/cm**3
    
    T : float
        temperature in units of K
    
    P : float
        pressure in units of hPa
    
    s : float
        slant distance in units of cm
        
        
    Methods:
    -------
    
    get_slant_distance : Calculate slant distance
    get_P_T : Interpolate pressure, temperature and rho
    """
    
    
    def __init__(self, x0, y0, h0=None, zenith=None, df=None, ds=None, steps_h=None):
        
        self.h0 = h0
        self.zenith = zenith
        self.df = df
        self.ds = ds
        self.steps_h = steps_h
        self.x0 = x0
        self.y0 = y0
        self.atmosphere = None        
        self.calc_atmo
        
        return 
    
    
    @property
    def calc_atmo(self):
        """
        Constructor of class Atmosphere.
        Calculate discretized atmosphere assuming spherical symmetry. 
        """
        if(self.x0 == 0.):
            #htop is hardcoded here - should be taken from df
            h = np.linspace(self.h0, 120, self.steps_h)

            # Calculate slant distance
            s = self.get_slant_distance(h, self.zenith)
            
        if(self.x0 < 0.):
            hprime = np.sqrt(self.x0**2+self.y0**2)/np.tan(np.deg2rad(self.zenith))
            h1 = np.linspace(self.h0 + hprime, 120, self.steps_h)
            h2 = np.linspace(self.h0 + hprime, self.h0, self.steps_h)
            
            s1 = self.get_slant_distance(h1, self.zenith)
            s2 = self.get_slant_distance(h2, 180. + self.zenith, False)
            s2 += s1[-1]
            
            s = np.concatenate([s1, s2])
            h = np.concatenate([h2[1:][::-1], h1])
            
        if(self.x0 > 0.):
            hprime = np.sqrt(self.x0**2+self.y0**2)/np.tan(np.deg2rad(self.zenith))
            h = np.linspace(self.h0 - hprime, 120, self.steps_h)
            
            s = self.get_slant_distance(h, self.zenith)
            
            idx = np.abs(h - self.h0).argmin()
            
            s = s[:(len(s) - idx)]
            h = h[idx:]

        # interpolation will go wrong for values ds < s[0]
        if(self.ds < s[0]):
            warnings.warn(f"""
                          ds is smaller than first path length for the current step size in h!
                          s[0] = {s[0]} - check interpolation of rho, T and P!
                          """)

        N_ds = int(s[-1]/(self.ds))
        model_s = np.cumsum(np.full(N_ds, self.ds))
        h = h[1:][::-1]
        # here interp_h will have units of km
        # and will start from top of the atmosphere
        interp_h = np.interp(model_s/1e5, s/1e5, h)

        # Now interpolate data from df 
        # Note that this interpolation will also give us the correct
        # values for large zenith angles, because we are abusing
        # spherical symmetry
        P, Pw, T, rho, delta = self.get_P_T_rho(interp_h)
        # Calculate slant depth by assuming rho is constant in one shell
        rho_top = self.df["rho"].values[-1]          
        rho_mid = rho[:-1] + 0.5 * np.diff(rho)
        
        dX_top = (rho_top + 0.5 * (rho[0] - rho_top)) * self.ds
        dX = rho_mid * self.ds
        dX = np.insert(dX, 0, dX_top)
        X = np.cumsum(dX)
        interp_h *= 1e5  #km -> cm
        
        self.atmosphere = np.vstack((dX, X, model_s, rho, interp_h, P, T, delta))
        
        return    
     
    
    @property
    def params2dict(self):
        resu = {}
        keys = ["dX", "X", "s", "rho", "h", "P", "T", "delta"]
        
        for i,key in enumerate(keys):
            resu[key] = self.atmosphere[i]
            
        return resu
    
    
    def get_slant_distance(self, h, zenith, reverse=True):
        """"
        Calculate slant distance between shell layers of atmosphere. This way the slant distance
        for large zenith angles (zenith > 70 deg) is also calculated correctly.
        Refraction is not included (TBD: https://www.ess.uci.edu/~cmclinden/link/xx/node45.html)
        Algorithm is also taken from the link above.
        
        Parameters:
        -----------
        h : numpy.ndarray
            height in km 
            
        Return:
        ------
        s : numpy.ndarray
            cumulative integrated slant distances for each spherical shell
            units: cm
        """
    
        R_E       = 6.3781e3 # Radius of earth in km
        zd        = np.deg2rad(zenith)
        sin_start = np.sin(zd)

        fraction = (R_E + h[1:])/(R_E + h[:-1])
        sin_frac = sin_start/np.cumprod(fraction)
        z_t      = (R_E + h[1:]) * sin_frac - R_E
        
        f1       = (R_E + h[1:])**2 - (R_E + z_t)**2
        f2       = (R_E + h[:-1])**2 - (R_E + z_t)**2

        s = abs(np.sqrt(f1) - np.sqrt(f2))
        
        if(reverse == True):
            s = s[::-1] * 1e5
        else:
            s *= 1e5

        resu = np.cumsum(s)

        return resu

    
    def get_P_T_rho(self, h):
        """"
        Interpolate the input data frame for pressure, temperature and mass density.
        Needed for fluorescence/cherenkov calculations.
        
        Parameters:
        -----------
        h : numpy.ndarray
            height in km
        
        Return:
        ------
        P : numpy.ndarray
            interpolated pressure in units of mbar
            
        T : numpy.ndarray
            interpolated temperature in units of K
        
        rho: numpy.ndarray
            interpolated mass density in units of g/cm^3
        """
            
        P  = np.interp(h, self.df["h"].values, self.df["P"].values)
        Pw = np.interp(h, self.df["h"].values, self.df["P"].values * self.df["Pw/P"])
        T  = np.interp(h, self.df["h"].values, self.df["T"].values)
        rho = np.interp(h, self.df["h"].values, np.log10(self.df["rho"].values))
        delta = np.interp(h, self.df["h"].values, np.log10(self.df["n-1"].values))
        
        return P, Pw, T, 10**rho, 10**delta
    
    
    @staticmethod
    def absorb_LUT(absorb_model):
        """
        Create look up table for atmospheric absorbtion. 
        Input is MODTRAN output as df.
        """
        
        h_model = np.array(absorb_model.columns[1:], dtype = float)  #first value is obs.lvl.
        data_model = absorb_model.to_numpy()
        
        h_interp = np.arange(h_model[0], h_model[-1], 0.1) #km (1.885km - 100km)
        wvl_interp = data_model[:,0].astype(int)           #wvl band in MODTRAN output (250nm-700nm)
        
        out = np.empty((len(wvl_interp), len(h_interp)))
        
        for i in range(len(wvl_interp)):
            out[i] = np.interp(h_interp, h_model, data_model[i][1:])    #first value in data_model is wvl
        
        return out, wvl_interp

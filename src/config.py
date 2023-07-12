import numpy as np
import pandas as pd
import glob
import os

from .atmosphere import Atmosphere
from .profile import Profile
from .lateraldistribution import LateralDistribution
from .tracking import Tracking
from .auxiliary import Point2Vector, Vec2Unit

class ConfigShower:
    
    def __init__(self, config = None, config_dict = None):
        
        if(config is not None):
            self.emission_type = config["general"]["emission"]
            self.toteff_model = config.getint("general", "toteff_model")
            if(ConfigShower.check_option(config, "general", "seed2")):
                self.wvl_seed = config.getint("general","seed2")
            if(ConfigShower.check_option(config, "general", "seed3")):
                self.pe_seed = config.getint("general", "seed3")
            
            self.x0 = config.getfloat("shower", "x0")
            self.y0 = config.getfloat("shower", "y0")
            self.theta = config.getfloat("shower", "theta")
            self.az = config.getfloat("shower", "az")
            self.get_corsika_data(config)

            self.atm_vertical = self.AtmPrf2df(config["atmosphere"]["atm_profile"])
            self.atm_absorbtion = self.AtmAbsorb2df(config["atmosphere"]["atm_absorb"])
            self.h0 = config.getfloat("atmosphere", "h0")
            self.ds = config.getint("atmosphere", "ds")
            self.steps_h = config.getint("atmosphere", "steps_h")

            self.tel_pos = self.calc_TelPos(config)
            self.tel_az = config.getfloat("telescope", "tel_az")
            self.tel_zenith = config.getfloat("telescope", "tel_zenith")
            self.R = config.getfloat("telescope", "R")
            self.FoV = config.getfloat("telescope", "FoV")
            self.pixel_shape = config.getint("telescope", "pixel_shape")
            self.focal_length = config.getfloat("telescope", "focal_length")
            self.pix_width = config.getfloat("telescope", "pix_width")
            self.camAltAz = self.calc_camPhiTheta(config["telescope"]["pixel_list"])

            self.NSB = config.getfloat("telescope", "NSB")
            self.int_time = config.getfloat("telescope", "integration_time")
            self.Npix = np.rint(self.FoV/self.pix_FoV).astype(int)
            self.qeff = self.Qeff2df(config["telescope"]["qeff"])
            self.mirror_ref = self.MirrorRef2df(config["telescope"]["mirror_ref"])
            self.camera_filter = self.CamFilter2df(config["telescope"]["camera_filter"])
            self.camera_trans = config.getfloat("telescope", "camera_trans")
            self.light_guide = config.getfloat("telescope", "light_guide_eff")
            self.pixel_saturation = config.getfloat("telescope", "pixel_saturation")
            self.psf = self.PSF2df(config["telescope"]["psf"])
            
            self.tmin = config.getint("binning", "tmin")
            self.tmax = config.getint("binning", "tmax")
            self.rbins = self.calc_LateralBins(config)
            self.Ebins = self.calc_EnergyBins(config)
            self.Nvoxel = config.getint("binning", "Nvoxel")

            self.Ecut = config.getboolean("fluorescence", "Ecut")
            self.wvl_lo = config.getfloat("cherenkov", "wvl_lo")
            self.wvl_up = config.getfloat("cherenkov", "wvl_up")
            self.MomAngBins = self.calc_MomentumAngleBins(config)
            
            self.tailcuts = np.array([config.getfloat("cleaning", "tailcut_hi"),
                                      config.getfloat("cleaning", "tailcut_lo")])

            self.__post_init__
            print(self)
            
        elif(config_dict is not None):
            for name,value in config_dict.items():
                setattr(self, name, value)
                
        else:
            raise Exception("Can not construct configure object!")

            
    @property
    def __post_init__(self):
        #calculate midpoints of energy and lateral distance bins
        self.calc_bins_mid
        
        #discretize spherical atmosphere and get parameter values
        spherical_atmosphere = Atmosphere(self.x0, self.y0, self.h0, self.theta,\
                                          self.atm_vertical, self.ds, self.steps_h)
        atmo_params = spherical_atmosphere.params2dict
        
        for key,val in atmo_params.items():
            setattr(self, "spherical_atmo_"+key, val)
            
        #get profile for tmin/tmax
        profile = Profile(self.X_in, self.N_in, self.spherical_atmo_X,\
                          self.spherical_atmo_h, self.tmin, self.tmax)
        profile_params = profile.params2dict
        
        for key,val in profile_params.items():
            setattr(self, "prf_"+key, val)
          
        #calculate lateral distance bins in units of Moliere radius
        self.x = LateralDistribution.calc_x(self.spherical_atmo_rho, self.prf_tmask, 
                                            self.rbins)
        
        #calculate coordinates along shower axis for tmin < t < tmax
        self.x_shower,\
        self.y_shower,\
        self.z_shower = Tracking.get_coords_shower_axis(self.theta, self.az,\
                                                        self.x0*1e5, self.y0*1e5,\
                                                        self.spherical_atmo_s,\
                                                        self.prf_tmask)
        
        #unit vector for shower axis (pointing upwards)
        point1 = np.array([self.x_shower[-1], self.y_shower[-1], self.z_shower[-1]])
        point2 = np.array([self.x_shower[0], self.y_shower[0], self.z_shower[0]])
        vector_shower = Point2Vector(point1, point2)
        self.unit_shower = Vec2Unit(vector_shower)
        
        return
       
        
    def AtmPrf2df(self, path):
        """
        Create dataframe from vertical atmosphere profile.
        path: str, path to atmospheric profile.
        """

        if(not os.path.isfile(path)):
            raise Exception('Invalid path to vertical atmospheric profile file: %s' % path)

        resu = pd.read_csv(path, skiprows = 3, delimiter= '\s+', index_col=False, 
                           names=['h', 'rho', 'X', 'n-1', 'T', 'P', 'Pw/P'])
        return resu

    
    def AtmAbsorb2df(self, path):
        """
        Create dataframe from absorbtion model - height and wavelength dependent
        path: str, path to absorbtion model
        """
        
        if(not os.path.isfile(path)):
            raise Exception('Invalid path to atmospheric absorbtion file: %s' % path)
            
        resu = pd.read_csv(path, delimiter= '\s+', index_col=False)
        
        return resu
    
    
    def Qeff2df(self, path):
        """
        Create dataframe from quantum efficiency file
        path = str, path to wavelength dependent quantum efficiency file
        """
        
        resu = pd.read_csv(path, skiprows = 15, usecols = [0,1], delimiter= '\s+', 
                           index_col=False, names=['wvl', 'qe'])
        
        return resu
    
    
    def MirrorRef2df(self, path):
        """
        Create dataframe from mirror reflectivity file
        path = str, path to wavelength dependent mirror reflectivity file
        """
        
        resu = pd.read_csv(path, skiprows = 1, usecols = [0,1], delimiter= '\s+', 
                           index_col=False, names=['wvl', 'reflectivity'])
        
        return resu
    
    
    def CamFilter2df(self, path):
        """
        Create dataframe from camera filter file
        path = str, path to wavelength dependent camera filter file
        """
        
        resu = pd.read_csv(path, skiprows = 3, usecols = [0,1], delimiter= '\s+', 
                           index_col=False, names=['wvl', 'eff'])
        
        return resu
    
    
    def PSF2df(self, path):
        """
        Create dataframe from 68% containment radius vs. off axis angle file
        path = str, path to PSF file
        """
        
        resu = pd.read_csv(path, skiprows = 1, delimiter='\s+', index_col=False,
                           names=['off_axis_angle', 'r68'])
        
        return resu 
    
    
    def calc_LateralBins(self, config):
        """ Calculate equally spaced lateral bins """

        rmin = config.getfloat("binning", "rmin")
        rmax = config.getfloat("binning", "rmax")
        rsteps = config.getfloat("binning", "rsteps")

        return np.arange(rmin, rmax, rsteps)


    def calc_EnergyBins(self, config):
        """ Calculate log. spaced energy bins """

        Emin = config.getfloat("binning", "Emin")
        Emax = config.getfloat("binning", "Emax")
        Ebins = config.getint("binning", "Ebins")

        return np.geomspace(Emin, Emax, Ebins)
    
    
    def calc_MomentumAngleBins(self, config):
        theta_min = config.getfloat("cherenkov", "theta_min")
        theta_max = config.getfloat("cherenkov", "theta_max")
        theta_bins = config.getint("cherenkov", "theta_bins")
        
        return np.linspace(theta_min, theta_max, theta_bins)
    
    
    def calc_TelPos(self, config):
        """ Return telescope position as numpy array"""
        
        x = config.getfloat("telescope", "x") * 1e5
        y = config.getfloat("telescope", "y") * 1e5
        z = config.getfloat("telescope", "z") * 1e5
        
        return np.array([x,y,z])
    
    
    def get_corsika_data(self, config):
        run = config.getint("shower", "run")
        shower = config.getint("shower", "shower")
        path = glob.glob(f"profile_lib/{run}/{shower}_*")
        data = np.load(path[0])
        corsika_xsteps = config.getfloat("shower", "corsika_xsteps")
        
        self.X_in = data["depth"]
        self.N_in = (data["electron"]+data["positron"])
        self.edep1 = data["EM"]*1e3/corsika_xsteps
        #Ecut data
        self.Edep_Ecut = data["EMCUT"]*1e3/corsika_xsteps
        self.X_in_Ecut = data["depth"] - 1.  #### NEED TO FIX THIS !!!! ####
        
        print("### Loaded profile for:")
        print(f"### zenith: {data['zenith']}")
        print(f"### energy: {data['energy']}")
        print(f"### height of first interaction: {data['height']}")
           
        return
    
    
    @property
    def calc_bins_mid(self):
        self.Ebins_mid = 10**(np.log10(self.Ebins[:-1]) + 0.5 * np.diff(np.log10(self.Ebins)))
        self.rbins_mid = self.rbins[:-1] + 0.5 * np.diff(self.rbins)
        
        return
    
    
    @staticmethod
    def set_impact(_config, i):
        """
        Set x,y impact position [km] in case of Nshower > 1
        """
        
        xscat = _config.getfloat("general", "xscat")
        yscat = _config.getfloat("general", "yscat")
        
        if(ConfigShower.check_option(_config, "general", "seed1")):
            seed = _config.getint("general", "seed1") + i
            rng = np.random.default_rng(seed = seed).random(size = 2)
        else:
            rng = np.random.default_rng().random(size = 2)
        
        x0 = 2. * xscat * rng[0]  - xscat
        y0 = 2.* yscat * rng[1] - yscat
        
        _config.set('shower', 'x0', f"{x0:.2f}")
        _config.set('shower', 'y0', f"{y0:.2f}")
                   
        return 
    
    
    @staticmethod
    def check_option(_config, section, option):
        """
        Check existence of option in section for config file _config
        return : bool
        """
        return _config.has_option(section, option)
            
    
    def calc_camPhiTheta(self, path):
        
        self.pix_FoV = np.rad2deg(2.*np.arctan(0.5*self.pix_width/self.focal_length))
        cm_to_deg = self.pix_FoV/self.pix_width
        pixel_list = np.loadtxt(path, skiprows=6, usecols=(3,4))
        
        self.CamAlt = pixel_list[:, 0] * cm_to_deg
        self.CamAz = pixel_list[:, 1] * cm_to_deg
        
        return
       

    def calc_camAltAz(self, path):
        """
        Transform camera offsets to corresponding Alt/Az values.
        """
        
        if(self.tel_az <= 180.):
            pointingAz = 180. - self.tel_az
        else:
            pointingAz = 360. - (self.tel_az - 180.)
        
        pointingAlt = 90. - self.tel_zenith
        
        pixel_list = np.loadtxt(path, skiprows=6, usecols=(3,4))
        self.CamAlt, self.CamAz = Projection.offset_to_angles(pixel_list[:,0], 
                                                              pixel_list[:,1], 
                                                              np.deg2rad(pointingAz), 
                                                              np.deg2rad(pointingAlt), 
                                                              self.focal_length)
        return
        
        
    def __str__(self):
        index = np.argmax(self.prf_N)
        Nmax_posx, Nmax_posy, Nmax_posz = self.x_shower[index], self.y_shower[index],\
                                          self.z_shower[index]
        
        Xmax_index = np.where(self.prf_t == 0.)[0][0]
    
        resu = "#### Starting with shower ####\n"\
        f"Shower age range: [{self.tmin}, {self.tmax}]\n"\
        "\n"\
        f"Energy range of particles: [{np.min(self.Ebins):.2e}, {np.max(self.Ebins):.2e}] MeV\n"\
        "\n"\
        "Minimum/Maximum lateral distance in units of meter\n"\
        f"{self.rbins_mid[:2]}...{self.rbins_mid[-2:]} m\n"\
        "Minimum/Maximum lateral distance in units of Moliere radius\n"\
        f"t = {np.min(self.prf_t):.2f}: {self.x[0][0]:.2e} / {self.x[0][-1]:.2e}\n"\
        f"t = 0:     {self.x[Xmax_index][0]:.2e} / {self.x[Xmax_index][-1]:.2e}\n"\
        f"t = {np.max(self.prf_t):.2f}: {self.x[-1][0]:.2e} / {self.x[-1][-1]:.2e}\n"\
        "\n"\
        f"Shower will be discretized in steps of {self.ds/100.} m along shower axis\n"\
        f"t       X[g/cm2]      S[km]     height[km]    rho[g/cm3]\n"\
        f"{np.min(self.prf_t):.2f}   {self.prf_X[0]:.2e}     {self.spherical_atmo_s[self.prf_tmask][0]/1e5:.2e}   {self.spherical_atmo_h[self.prf_tmask][0]/1e5:.2e}      {self.spherical_atmo_rho[self.prf_tmask][0]:.2e}\n"\
        f"0.00    {self.prf_X[Xmax_index]:.2e}     {self.spherical_atmo_s[self.prf_tmask][Xmax_index]/1e5:.2e}   {self.spherical_atmo_h[self.prf_tmask][Xmax_index]/1e5:.2e}      {self.spherical_atmo_rho[self.prf_tmask][Xmax_index]:.2e}\n"\
        f"{np.max(self.prf_t):.2f}   {self.prf_X[-1]:.2e}     {self.spherical_atmo_s[self.prf_tmask][-1]/1e5:.2e}   {self.spherical_atmo_h[self.prf_tmask][-1]/1e5:.2e}      {self.spherical_atmo_rho[self.prf_tmask][-1]:.2e}\n"\
        "\n"\
        f"Shower is coming from azimuth/zenith: {self.az}/{self.theta} deg\n"\
        f"Telescope pointing azimuth/zenith: {self.tel_az}/{self.tel_zenith} deg\n"\
        "\n"\
        "Impact position of shower (x,y,z) in km:\n"\
        f"({self.x0:.2f}, {self.y0:.2f}, {self.spherical_atmo_h[-1]/1e5:.2f})\n"\
        "Position of shower maximum in km:\n"\
        f"({Nmax_posx/1e5:.2f}, {Nmax_posy/1e5:.2f}, {Nmax_posz/1e5:.2f})\n"\
        f"Distance of shower maximum to telescope position: {np.linalg.norm([Nmax_posx, Nmax_posy, Nmax_posz])/1e5:.2f} km\n"
        
        return resu

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import HTML
from matplotlib.animation import ArtistAnimation
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from numba import set_num_threads

from .atmosphere import Atmosphere
from .telescope import Telescope
from .tracking import Tracking
from .ctypes_wrapper import c_arrival_times, c_apply_qeff_absorbtion
from .cherenkov import Cherenkov
from .projection import Projection
from .imageparams import ImageParams
from .auxiliary import plot_iact_image, numba_apply_qeff_abs
from .multi_threading_params import _NUM_THREADS_CHER


class Analysis:
    """
    Image, Signal, Emissivity etc.
    TBD
    """
    
    def __init__(self, Nph = None, idx_Ch = None, cfg = None,
                 path_to_data = None, event_type = None):
            
        self.idx_Ch = None
        self.Npe = None
        self.Nph = None
        self.CamSig = None
        self.theta = None
        self.phi = None
        self.wvl_bins = None
        self.absorb_LUT = None
        self.FoV = None
        self.Npix = None
        self.s = None
        self.tel_pos = None
        self.tel_az = None
        self.tel_zenith = None
        self.int_time = None
        self.NSB = None
        self.wvl_seed = None
        self.pe_seed = None
        self.psf = None
        
        if(path_to_data is not None):
            loaded = np.load(path_to_data, allow_pickle=True)
            cfg = ConfigShower(config_dict = loaded["config"][()])
            if(cfg.emission_type == "both"):
                if(event_type is None):
                    raise KeyError("Please provide an event type")
                if(event_type == "cherenkov"):
                    self.Nph = loaded["Nph_Ch"]
                    self.idx_Ch = loaded["idx_Ch"]
                else:
                    self.Nph = loaded["Nph_Fl"]

            elif(cfg.emission_type == "cherenkov"):
                self.Nph = loaded["Nph"]
                self.idx_Ch = loaded["idx_Ch"]

            else:
                self.Nph = loaded["Nph"]
    
        else:
            self.Nph = Nph
            if(idx_Ch is not None):
                self.idx_Ch = idx_Ch
            cfg = cfg
            
        self.dimensions = (len(cfg.prf_t), len(cfg.rbins_mid), cfg.Nvoxel)
        self.s = cfg.spherical_atmo_s[cfg.prf_tmask]
        self.tel_pos = cfg.tel_pos
        self.tel_az = cfg.tel_az
        self.tel_zenith = cfg.tel_zenith
        self.FoV = cfg.FoV
        self.pixel_shape = cfg.pixel_shape
        self.pix_FoV = cfg.pix_FoV
        self.CamAlt = cfg.CamAlt
        self.CamAz = cfg.CamAz
        self.Npix = cfg.Npix
        self.int_time = cfg.int_time
        self.NSB = cfg.NSB
        self.tailcuts = cfg.tailcuts
        self.psf = cfg.psf
        self.pixel_saturation = cfg.pixel_saturation
        
        if(hasattr(cfg, "wvl_seed")):
            self.wvl_seed = cfg.wvl_seed
        if(hasattr(cfg, "pe_seed")):
            self.pe_seed = cfg.pe_seed
        
        self.absorb_LUT, self.wvl_bins = Atmosphere.absorb_LUT(cfg.atm_absorbtion)
        self.eff_interp = Telescope.interp_qeff(self.wvl_bins, cfg.qeff, cfg.mirror_ref,
                                                cfg.camera_filter)
        self.eff_interp *= cfg.camera_trans * cfg.light_guide
     
        self.pipe(cfg)
         
        
    def pipe(self, cfg):
        """
        Calculate arrival times, camera coordinates of voxel positions and apply
        atmospheric absorbtion and quantum efficiency.
        Finally calculate p.e. values for camera pixel list.
        """
    
        Track = Tracking(cfg)
        coords = Track.get_coords
        self.calc_arrival_times(coords)
       
        if(self.idx_Ch is not None):
            coords = coords[self.idx_Ch]
            self.time_lag = self.time_lag[self.idx_Ch]
            
        self.apply_qeff_absorbtion(cfg, coords)
        
        self.theta, self.phi = Projection.get_camera_spherical(self.tel_pos,
                                                               self.tel_az,
                                                               self.tel_zenith,
                                                               coords)
                
        self.fov_mask = self.get_FoV_mask
        if(self.fov_mask.size == 0):
            raise ValueError("No photons in FoV")

        #Include PSF
        self.PSF_smearing
        
        #Calculate camera signal
        indices, missing_neighbour_index = self.get_pixel_id
        missing_neighbours = indices != missing_neighbour_index
        self.CamSig = np.bincount(indices[missing_neighbours], 
                                  weights=self.Npe[missing_neighbours], 
                                  minlength=len(self.CamAlt))
        
        return
    
    
    @property
    def PSF_smearing(self):
        """
        Incorporate optics PSF by smearing out coordinates of voxels in the telescope
        coordinate system. 
        """
        
        off_axis_interp = interp1d(self.psf["off_axis_angle"].values, 
                                   self.psf["r68"].values, 
                                   bounds_error=False, fill_value="extrapolate")
        #estimate of cog_x
        cogx = np.sum(self.theta[self.fov_mask] *
                      self.Npe[self.fov_mask])/np.sum(self.Npe[self.fov_mask])
        print(f"Estimated cog_x :{cogx:.2f}")
        #r68 value at cog_x defines the PSF for simplicity
        mean = (0, 0)
        #assume symmetric 2D gaussian
        sigma = off_axis_interp(abs(cogx))/np.sqrt(np.log(1./0.32)*2)
        smearing = np.random.default_rng().multivariate_normal(mean, 
                                                               [[sigma**2, 0], 
                                                                [0, sigma**2]], 
                                                                size = self.theta.shape[0]
                                                              )
        
        self.theta += smearing[:, 0]
        self.phi   += smearing[:, 1]
        
        return
    
    
    @property
    def get_pixel_id(self):
        "Calculate the pixel_id of photons hitting the sphere"
        
        midpoints = np.squeeze(np.dstack((self.CamAlt, self.CamAz)))
        hits = np.squeeze(np.dstack((self.theta, self.phi)))
        
        # Construct a KDTree of the midpoint coordinates
        kdtree = cKDTree(midpoints)
        
        # Query the KDTree to get the closest midpoint to each hit
        if(self.pixel_shape == 0):
            indices = kdtree.query(hits, k = 1, workers = _NUM_THREADS_CHER,
                                   distance_upper_bound = self.pix_FoV/np.sqrt(2.))[1]
        elif(self.pixel_shape == 1):
            indices = kdtree.query(hits, k = 1, workers = _NUM_THREADS_CHER,
                                   distance_upper_bound = self.pix_FoV/2.)[1]
        elif(self.pixel_shape == 2):
            indices = kdtree.query(hits, k = 1, workers = _NUM_THREADS_CHER,
                                   distance_upper_bound = self.pix_FoV/np.cos(np.deg2rad(30.)))[1]
        else:
            raise ValueError("Pixel shape not supported!")
        
        return indices, kdtree.n  
    
    
    @property
    def get_FoV_mask(self):
        """
        Create a boolean array to distinguish between voxels inside and outside
        of FoV. This is mainly an issue for the fluorescence case. Need to figure out
        something better.
        """
        
        half_FoV = self.FoV/2.
        FoV_mask = ((abs(self.theta) < half_FoV) & (abs(self.phi) < half_FoV))
        
        return FoV_mask
    
    
    def Nph_in_FoV(self, Nph):
        print(f"Total number of fluorescence photons in FoV: {np.sum(Nph[self.FoV_mask]).astype(int)}")
        print(f"Total number of fluorescence photons outside FoV: {np.sum(Nph[~self.FoV_mask]).astype(int)} \n")
    
        return
    
    
    def calc_arrival_times(self, coords):
        """
        Calculate arrival times of photons in seconds.
        Arrival time = time lag along shower axis + dist_to_telescope/c
        """
        
        c = 299792458 * 1e2  # cm/s
        
        tel_dist = Telescope.calc_rel_dist(self.tel_pos, coords)
        
        # Calculate time lag along shower axis
        axis_time_lag = self.s/c 
        # Total time lag
        self.time_lag = c_arrival_times(axis_time_lag, tel_dist)
        
        return 
    
    
    def apply_qeff_absorbtion(self, cfg, coords):
        #draw random wavelength for each voxel - this means all photons inside
        #a single voxel will have the same wavelength.
        if(cfg.toteff_model == 0):
            print("Drawing random wavelengths for each voxel.\n"\
                  "Care: all photons inside a single voxel will have the same wavelengths!")
            if(self.idx_Ch is not None):
                #draw random wvl for cherenkov photons
                wvl_idx = Cherenkov.draw_random_wvl(cfg.wvl_lo, 
                                                    cfg.wvl_up, 
                                                    self.Nph.shape[0], 
                                                    self.wvl_bins[0])

                self.Npe = c_apply_qeff_absorbtion(self.absorb_LUT,
                                                   wvl_idx.astype("int32"),
                                                   self.eff_interp,
                                                   coords,
                                                   self.Nph, cfg)

            else:
                #draw random wvl for fluorescence photons
                wvl_idx = Fluorescence.draw_random_wvl(self.Nph.shape[0], 
                                                       self.wvl_bins)

                self.Npe = c_apply_qeff_absorbtion(self.absorb_LUT,
                                                   wvl_idx,
                                                   self.eff_interp, 
                                                   coords,
                                                   self.Nph, cfg)
                
        #overall efficiency - no random wavelengths drawn. This method should be used
        #if the shower binning is coarse. However, the computation time with this method
        #is a bit longer. 
        else:
            distances = Telescope.calc_rel_dist(self.tel_pos, coords)
            scaling = (coords[:, 2] - self.tel_pos[2])/distances
            height_idx = ((coords[:, 2]/1e5 + cfg.h0)/0.1).astype(int)
            set_num_threads(_NUM_THREADS_CHER)
            #cherenkov case
            if(self.idx_Ch is not None):
                wvl_mask = np.where((self.wvl_bins >= cfg.wvl_lo) &
                                    (self.wvl_bins <= cfg.wvl_up))[0]
                one_over_wvl_sqr = (1./self.wvl_bins[wvl_mask])**2
                denom = np.trapz(one_over_wvl_sqr, self.wvl_bins[wvl_mask])

                wvl = self.wvl_bins[wvl_mask].copy()
                eta = one_over_wvl_sqr * self.eff_interp[wvl_mask]

                args = (self.Nph, height_idx, self.absorb_LUT, 
                        wvl_mask, scaling, denom, eta, cfg.emission_type, wvl)

                self.Npe = numba_apply_qeff_abs(*args)
            #fluorescence case 
            else:
                wvl_Fl = np.array(Fluorescence.model)[:,0]
                wvl_mask = np.searchsorted(self.wvl_bins, wvl_Fl).astype(int)
                
                prob_Fl = np.array(Fluorescence.model)[:,1]
                denom = np.sum(prob_Fl)
                eta = prob_Fl * self.eff_interp[wvl_mask]

                args = (self.Nph, height_idx, self.absorb_LUT, 
                        wvl_mask, scaling, denom, eta, cfg.emission_type)

                self.Npe = numba_apply_qeff_abs(*args)

        return
    
    
    def show_signal(self, _int_time = None):
        "Plot of signal, integration time in microseconds"
        
        if(_int_time is not None):
            int_time = _int_time
        else:
            int_time = self.int_time
            
        FoV_mask = self.get_FoV_mask
        
        time_dat = self.time_lag[FoV_mask]
        time_dat -= np.min(self.time_lag[FoV_mask])
        time_dat *= 1e6
        time_bin = np.arange(np.min(time_dat), np.max(time_dat), int_time)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        
        ax.hist(time_dat, bins = time_bin, 
                weights = self.Npe[FoV_mask]/int_time);
            
        ax.axes.xaxis.set_label_text(r"Time [$\mu$s]")
        ax.axes.yaxis.set_label_text(r"N_pe/$\mu$s")
        
        return ax
    
    
    @property
    def image(self):
        """
        Calculate image and Hillas parameters.
        """
        
        int_time = (np.max(self.time_lag) - np.min(self.time_lag)) * 1e6 #s->microseconds
        NSB = int_time * self.NSB * 1e3  # 1/ns -> 1/microseconds
        
        print(f"Expected mean p.e./pixel/{int_time:.2f}\u03BCs = {NSB:.2f}")
        
        if(self.pe_seed is not None):
            sig = np.random.default_rng(seed = self.pe_seed).poisson(self.CamSig+NSB, 
                                                                     size=self.CamSig.shape)
        else:
            sig = np.random.default_rng().poisson(self.CamSig+NSB, size=self.CamSig.shape)
        
        self.CamSig[self.CamSig > self.pixel_saturation] = self.pixel_saturation
        
        image_params = ImageParams(self.CamSig, [self.CamAz, self.CamAlt], 
                                   self.tailcuts)
        self.Hillas = image_params.params2dict
        
        return sig 
     
        
    def show_image(self, ellipse = False, *args, **kwargs):
        """"
        Plot image of signal sum
        
        Alt, x
        ^
        |
        |
        |
        |
        |---------->Az, y
        """
        
        sig = self.image
        
        fig, axs = plt.subplots(facecolor="white")
        axs.grid(False)
        
        p = plot_iact_image(np.log10(np.maximum(sig, 1e-20)), 
                            self.CamAz, self.CamAlt,
                            self.pixel_shape,
                            sizehex=self.pix_FoV, 
                            lim = self.FoV/2. + 0.4, 
                            #orient = np.deg2rad(90.), 
                            ax = axs,
                            *args, **kwargs)

        axs.set_xlabel("Phi [deg]")
        axs.set_ylabel("Theta [deg]")
        
        fig.colorbar(p, ax=axs, label = "log10(p.e.)")
        if (ellipse):
            plot_ellipse(self.Hillas["cog_y"],
                         self.Hillas["cog_x"], 
                         self.Hillas["length"], 
                         self.Hillas["width"], 
                         self.Hillas["phi"]
                        )   
        plt.show() 
        
        return p
    
    
    def show_animated(self, _int_time = None, *args, **kwargs):
        "Plot animated image of signal with frame = time_reso"
        
        if(_int_time is None):
            int_time = self.int_time
        else:
            int_time = _int_time
        
        times = self.time_lag.ravel() * 1e6  #s->microseconds
        time_bins = np.arange(np.min(times), np.max(times), int_time)
        index_times = np.digitize(times, time_bins) - 1
        
        NSB = int_time * self.NSB * 1e3   #1/ns -> 1/microseconds
        
        print(f"Expected mean p.e./pixel/{int_time}\u03BCs = {NSB}")
        
        def plot_frame(CamAlt, CamAz, theta, phi, Npe, FoV, pixFoV, pixel_shape, NSB=NSB,
                       *args, **kwargs):
            
            
            midpoints = np.squeeze(np.dstack((CamAlt, CamAz)))
            hits = np.squeeze(np.dstack((theta, phi)))
        
            # Construct a KDTree of the midpoint coordinates
            kdtree = cKDTree(midpoints)

            # Query the KDTree to get the closest midpoint to each hit
            if(pixel_shape == 0):
                indices = kdtree.query(hits, k = 1, workers = _NUM_THREADS_CHER,
                                       distance_upper_bound = pixFoV/np.sqrt(2.))
            elif(pixel_shape == 1):
                indices = kdtree.query(hits, k = 1, workers = _NUM_THREADS_CHER,
                                       distance_upper_bound = pixFoV/2.)
            elif(pixel_shape == 2):
                indices = kdtree.query(hits, k = 1, workers = _NUM_THREADS_CHER,
                                       distance_upper_bound = pixFoV/np.cos(np.deg2rad(30.)))
            else:
                raise ValueError("Pixel shape not supported!")

            missing_neighbour_index = kdtree.n
            missing_neighbours = indices != missing_neighbour_index
        
            CamSig_trace = np.bincount(indices[missing_neighbours], 
                                       weights = np.squeeze(Npe[missing_neighbours]), 
                                       minlength = len(CamAlt))
            
            sig = np.random.default_rng().poisson(CamSig_trace+NSB, 
                                                  size = CamSig_trace.shape)
            
            title = plt.text(0.5, 1.01, f"{np.max(sig):.2f} max. p.e./pixel/{int_time}$\mu$s", 
                             ha="center", va="bottom", transform=axs.transAxes, fontsize="large")
           
            pp = plot_iact_image(np.log10(np.maximum(sig, 1e-20)), 
                                 CamAz, CamAlt,
                                 pixel_shape,
                                 sizehex = self.pix_FoV, 
                                 lim = FoV/2. + 0.4, 
                                 #orient = np.deg2rad(90.), 
                                 ax = axs,
                                 *args, **kwargs)
            
            return pp, title, sig
        
        
        fig, axs = plt.subplots()
        axs.grid(False)
        axs.set_xlabel("Phi [deg]")
        axs.set_ylabel("Theta [deg]")
       
        images = []
        for frame in np.arange(0, len(time_bins), 1):   
            index = index_times == frame
            im,title,sgnl = plot_frame(self.CamAlt, self.CamAz,
                                       self.theta[index], self.phi[index], 
                                       self.Npe[index],
                                       self.FoV, self.pix_FoV,
                                       self.pixel_shape,
                                       *args, **kwargs)
            images.append([im, title])
                    
        fig.colorbar(im, ax=axs, label = "p.e.")    
        ani = ArtistAnimation(fig, images, interval=100, blit=False,
                                        repeat_delay=500)
        
        return HTML(ani.to_jshtml())

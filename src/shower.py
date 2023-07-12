import numpy as np
from matplotlib import pyplot as plt
import os
import configparser

from .config import ConfigShower
from .energyspectrum import EnergySpectrum
from .lateraldistribution import LateralDistribution
from .tracking import Tracking
from .cherenkov import Cherenkov 

class Shower:
    """
    Base class to run the simulation code.
    Shower.run(path_config) will either 
    
    return 
    - an array with the number of photons hitting the sphere for each shower age, lateral
      distance and voxel position around the shower axis 
    - the config object 
    
    **or** 
    
    save this array and config object as .npz file.
    
    path_config: str, path to config file
    """
    
    def run(path_config=None, make_plots = False):
        if(path_config is None):
            return Shower.get_Nph(cfg, make_plots), cfg
        
        if(not os.path.isfile(path_config)):
            raise Exception('Invalid path to configuration file: %s' % path_config)
            
        _cfg = configparser.ConfigParser()
        _cfg.read(path_config)
        Nshower = _cfg.getint("general", "Nshower")
        
        if(Nshower > 1):
            for i in range(Nshower):
                ConfigShower.set_impact(_cfg, i)
                cfg = ConfigShower(config = _cfg)
                Shower.save_data(Shower.get_Nph(cfg, make_plots), _cfg, cfg, i)
        else:
            cfg = ConfigShower(config = _cfg)
            return Shower.get_Nph(cfg, make_plots), cfg
        
        return
    
    
    def get_Nph(cfg, make_plots):
        """
        Calculate number of photons hitting the sphere.
        If both - Cherenkov and Fluorescence photons- are calculated the return is
        Cherenkov photons, Fluorescence photons
        """
        
        Edist = EnergySpectrum(cfg)
        Lat = LateralDistribution(cfg)
        Track = Tracking(cfg)

        NlogE = Edist.get_Edist
        xSamples = Lat.get_xSamples(NlogE) 
        coords = Track.get_coords

        if(cfg.emission_type == "both"):
            Nph_Ch, idx = Shower.calc_CherenkovPhotons(cfg, coords, xSamples, make_plots)
            Nph_Fl = Shower.calc_FluorescencePhotons(cfg, coords, xSamples, make_plots)
            
            Shower.__del__((NlogE, xSamples, coords, Edist, Lat, Track))
            return Nph_Ch, idx, Nph_Fl
        
        elif(cfg.emission_type == "cherenkov"):
            Nph, idx = Shower.calc_CherenkovPhotons(cfg, coords, xSamples, make_plots)
            
            Shower.__del__((NlogE, xSamples, coords, Edist, Lat, Track))
            return Nph, idx
        
        elif(cfg.emission_type == "fluorescence"):
            Nph = Shower.calc_FluorescencePhotons(cfg, coords, xSamples, make_plots)
            
        else:
            raise KeyError("Invalid emission type: %s", cfg.emission_type)

        Shower.__del__((NlogE, xSamples, coords, Edist, Lat, Track))
        
        return Nph

    
    def calc_CherenkovPhotons(cfg, coords, xSamples, make_plots):
        
        Cher = Cherenkov(cfg)
        Nph_sphere = Cher.calc_Nph(coords, xSamples)
        coords_idx = np.where(Nph_sphere > 0.)[0]
        
        if(make_plots):
            #sum over energy axis and distribute photons equally over voxels
            #around shower axis
            Nph_tot = np.repeat((np.sum(Cher._Nch, axis = 1)/cfg.Nvoxel), cfg.Nvoxel)
            Shower.slice_along_shower(coords, Nph_tot, Nph_sphere)
            Shower.slice_perpendicular_shower(cfg, Nph_tot, Nph_sphere)
        
        return Nph_sphere[coords_idx], coords_idx


    def calc_FluorescencePhotons(cfg, coords, xSamples, make_plots):
        
        Fl = Fluorescence(cfg, xSamples)
        Nph_sphere = Fl.calc_Nph(coords, xSamples)
        
        if(make_plots):
            Nph_tot = np.repeat((Fl.Nph_shell/cfg.Nvoxel), cfg.Nvoxel)
            Shower.slice_along_shower(coords, Nph_tot, Nph_sphere)
            Shower.slice_perpendicular_shower(cfg, Nph_tot, Nph_sphere)
        
        return Nph_sphere
    
    
    def slice_along_shower(coords, Nph_tot, Nph_sphere):
        
        fig, axs = plt.subplots(1,2)
        axs[0].grid(False)
        axs[1].grid(False)
        
        pp1 = axs[0].hist2d(coords[..., 1].ravel()/1e5,
                            coords[..., 2].ravel()/1e5,
                            bins = 100,
                            weights = Nph_tot.ravel());
        
        
        pp2 = axs[1].hist2d(coords[..., 1].ravel()/1e5,
                            coords[..., 2].ravel()/1e5,
                            bins = 100,
                            weights = Nph_sphere.ravel());
        
        axs[0].set_title("Total photons")
        axs[1].set_title("Sphere photons")
        axs[0].set_xlabel("y [km]")
        axs[0].set_ylabel("z [km]")
        axs[1].set_xlabel("y [km]")
        axs[1].set_ylabel("z [km]")
        
        fig.colorbar(pp1[3], ax=axs[0], orientation='horizontal')
        fig.colorbar(pp2[3], ax=axs[1], orientation='horizontal')
        plt.show()
        
        return
     
        
    def slice_perpendicular_shower(cfg, Nph_tot, Nph_sphere):
        
        dim = (len(cfg.prf_t), len(cfg.rbins_mid), cfg.Nvoxel)
        rbins = cfg.rbins_mid
        abins = np.linspace(0., 2.*np.pi, cfg.Nvoxel, endpoint=False)
        A, R = np.meshgrid(abins, rbins)

        fig, axs = plt.subplots(1,2, subplot_kw=dict(projection="polar"))
        axs[0].grid(False)
        axs[1].grid(False)
        
        pc1 = axs[0].pcolormesh(A, R, np.sum(Nph_tot.reshape(dim), axis = 0),
                                cmap="magma_r", norm = LogNorm())

        pc2 = axs[1].pcolormesh(A, R, np.sum(Nph_sphere.reshape(dim), axis = 0), 
                                cmap="magma_r", norm = LogNorm())

        axs[0].set_yticklabels([])
        axs[1].set_yticklabels([])
        axs[0].set_title("Total photons")
        axs[1].set_title("Sphere photons")
        
        fig.colorbar(pc2, ax=axs[1], orientation='horizontal')
        fig.colorbar(pc1, ax=axs[0], orientation='horizontal')
        plt.show()
        
        return
    
    
    def save_data(data, _cfg, cfg, i):
        if(not ConfigShower.check_option(_cfg, "general", "save_dir")):
            raise KeyError("Nshower > 1 but no save directory provided!")
            
        path = _cfg["general"]["save_dir"] + f"Event{i}.npz"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if(cfg.emission_type == "both"):
            with open(path, 'wb') as f:
                np.savez(f, Nph_Ch = data[0], idx_Ch = data[1], Nph_Fl = data[2],
                         config=vars(cfg))
        elif(cfg.emission_type == "cherenkov"):
            with open(path, 'wb') as f:
                np.savez(f, Nph = data[0], idx_Ch = data[1], config=vars(cfg))
        else:
            with open(path, 'wb') as f:
                np.savez(f, Nph = data, config=vars(cfg))
                
        f.close()
        return
    
    
    def __del__(*elements):
        del(elements)

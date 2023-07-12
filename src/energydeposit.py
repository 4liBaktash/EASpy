import numpy as np
from matplotlib import pyplot as plt


class EnergyDeposit:
    """
    Calculates the contribution from electrons and positrons to the total deposit energy per slant depth at
    every shower stage.
    Stopping power is calculated using the Møller cross-section for electrons and Bhaba for positrons.
    Density effect is calculated by using Sternheimer's parametrization with coefficients for dry air.
    Deposit energy per slant depth is calculated in MeV cm^2/g.
    
    Attributes:
    ----------
    
    dEdX_perE: Dictionary with keys = electron, positron
               Each key contains a 2D numpy.ndarray which describes the
               contribution for each energy to the total energy deposit.
               dEdX_perE.shape = (len(t), len(E) - 1) 
               units: MeV * cm^2/g
               
    dEdX_tot : 1D numpy.ndarray
               total energy deposit for every evolution stage.
               units: MeV * cm^2/g
               
               
    Methods:
    -------
    bethe_bloch: calculates ionization losses for positrons and electrons at each evolution stage
    show_dEdX : plot the ionization losses for electrons and positrons as a function of energy
    show_Edep : plot the energy deposit as a function of slant depth
    """
    
    def __init__(self, cfg, xSamples):
           
        self.dEdX_tot = None
        self.dEdX_perE = None
        self.Ebins_mid = (cfg.Ebins_mid).copy()    
        self.t = (cfg.prf_t).copy()
        self.prf_X = (cfg.prf_X).copy()
        
        self.get_Edep(LateralDistribution.get_NperE(xSamples))
        
    
    def bethe_bloch(self, particle):
        """ 
        Calculates ionization losses in MeV cm**2/g.
        
        Parameters:
        -----------
        particle : string
                   electron or positron
                   if electron : Møller cross section
                   if positron : Bhabha cross section 
                   
        Return: 
        -------
        1D numpy.ndarray
        dEdX for a given particle type in units of MeV * cm^2/g
        """
        
        # Take mid energy points
        energy = self.Ebins_mid
        
        # Taken from Particle Data Group for dry air
        me = 0.510999                                # mass of electron in MeV
        K  = 0.307075                                # MeV cm2/mol
        I  = 85.7                                    # mean excitation energy in eV
        ZA = 0.49919                                 # mean atomic number over atomic mass

        gamma  = energy/me + 1.                      # E_kin = me(gamma - 1)                            
        beta   = np.sqrt(1. - 1./(gamma**2)) 

        # calculate ionization energy losses with Møller and Bhabha cross section
        # Formulas taken from 
        #"Stopping Powers and Ranges of Electrons and Positrons" - 
        # M.J. Berger and S.M. Seltzer
        if(particle == "electron"):
            dEdX    = np.log((energy/(I/1e6))**2) +  np.log(1. + (gamma - 1.)/2.) 
            dEdX   += (1. - beta**2) *  (1. + (gamma - 1.)**2/8. -\
                                         (2*(gamma - 1.) + 1.)*np.log(2))
            dEdX   -= _sternheimer_den(gamma, beta)
            dEdX   *= 0.5 * K * ZA * 1./(beta**2)

        if(particle == "positron"):
            dEdX     = np.log((energy/(I/1e6))**2) +  np.log(1. + (gamma-1.)/2.)
            dEdX    += 2. * np.log(2)
            dEdX    -= (beta**2/12.) * (23. + 14./((gamma-1.)+2.) +\
                                        10./(((gamma-1.)+2.)**2) +\
                                        4./(((gamma-1.)+2.)**3))
            dEdX    -= _sternheimer_den(gamma, beta)
            dEdX    *= 0.5 * K * ZA * 1./(beta**2)

        return dEdX
    
    
    @property
    def show_dEdX(self):
        """
        Show the ionization losses for electrons and positrons as a 
        function of energy.
        -------
        ax : AxesSubplot
        """
       
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        
        ax.plot(self.Ebins_mid, self.bethe_bloch("electron"), 
                'r--', label = "electron")
        ax.plot(self.Ebins_mid, self.bethe_bloch("positron"), 
                'b--', label = "positron")
        ax.axes.set_yscale("log")
        ax.axes.set_xscale("log")
        ax.axes.yaxis.set_label_text("dE/dX [MeV cm$^{2}$/g]")
        ax.axes.xaxis.set_label_text("Energy [MeV]")
        ax.axes.legend()
        
        plt.show()
        return  ax
    
    
    @property
    def show_Edep(self):
        """
        Show energy deposit as a function of slant depth.
        -------
        ax : AxesSubplot
        """
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        
        ax.plot(self.prf_X, np.sum(self.dEdX_perE["electron"], axis = 1),
                'r--', label = "electron")
        ax.plot(self.prf_X, np.sum(self.dEdX_perE["positron"], axis = 1), 
                'b--', label = "positron")
        ax.plot(self.prf_X, self.dEdX_tot, color = "black", ls = "solid",
                label = "total")
        
        ax.axes.set_yscale("log")
        ax.axes.yaxis.set_label_text("dE/dX [MeV cm$^{2}$/g]")
        ax.axes.xaxis.set_label_text("X [g/cm$^{2}$]")
        ax.axes.legend()
        
        plt.show()
        return  ax
    
    
    def get_Edep(self, NperE):
        """
        Note that we already have integrated LateralDistribution object over 
        log(x) and log(energy) - so only summing up the contributions
        for each x at fixed energy will give us the number of particles for
        each energy - see LateralDistribution.get_NperE()
        """
        
        self.dEdX_perE = {}

        for key in NperE.keys():
            # calculate ionization loss for a single particle with E = energy
            dEdX_particle = self.bethe_bloch(key)
            # contribution to total energy deposit from each energy and particle type:
            self.dEdX_perE[key] = NperE[key] * dEdX_particle[np.newaxis, :]

            N_tot = np.sum(NperE[key], axis = 1)
            if(key == "electron"):
                print(f"Maximum number of electrons: {np.max(N_tot)/1e5:.2f} x 1e5")
                print(f"... reached at t = {self.t[np.argmax(N_tot)]:.2f}")
            else:
                print(f"Maximum number of positrons: {np.max(N_tot)/1e5:.2f} x 1e5")
                print(f"... reached at t = {self.t[np.argmax(N_tot)]:.2f}")

       
        #total energy deposit per slant depth
        self.dEdX_tot  = np.sum(self.dEdX_perE["electron"], axis = 1)
        self.dEdX_tot += np.sum(self.dEdX_perE["positron"], axis = 1) 
        print(f"Maximum energy deposit: {np.max(self.dEdX_tot)/1e3:.2f} GeV")
        print(f"... reached at t = {self.t[np.argmax(self.dEdX_tot)]:.2f} ")

        return 
    

    
################################################################################
######################### Subroutines ##########################################
################################################################################
def _sternheimer_den(gamma, beta):
    """
    Calculates the density effect correction using Sternheimers parametrization.
    Coefficients taken from Particle Data Group.
    """
    
    Cbar = 10.591
    x0   = 1.7418
    x1   = 4.2759
    a    = 0.10914
    k    = 3.3994
    
    x = np.log10(gamma * beta)    
    
    delta = np.piecewise(x, [x >= x1, ((x0<=x) & (x<x1)), x < x0],
                        [lambda x: 2.*np.log(10)*x - Cbar, 
                         lambda x: 2.*np.log(10)*x - Cbar + a*(x1 - x)**k, 
                         lambda x: 0.]
                        )
    
    return delta

import numpy as np

class Ecut:
    "TBD - Calculate number of fluorescence photons falling below energy cut of corsika"

    @staticmethod
    def calc_Nphotons(Edep, Temp, P, ds, rho):
        "Calculate number of photons for Ecut data"
        normed = Fluorescence.get_normalized_model(Temp, P)
        Y0_337 = Fluorescence.Y0_337
        dX = ds * rho
        Nphotons = Edep * Y0_337 * normed * dX
        
        print(f"Total number of produced fluorescence photons with ecut data: {np.sum(Nphotons):.2e}")
        return Nphotons
    
    
    @staticmethod
    def Nph_per_shell(Edep, age, x, Temp, P, ds, rho):
        "Calculate number of fluorescence photons for each cylindrical shell"
        
        Nphotons = Ecut.calc_Nphotons(Edep, Temp, P, ds, rho)
        pdf = LateralDistribution.calc_pdf2D_x(age, x)
        #calculate probabilities
        cdf = np.diff(cumtpz(pdf, np.log(x), axis = 1, initial = 0.), axis = 1)
        #multiply with number of photons
        Nph_shell = Nphotons[:, np.newaxis] * cdf
        
        return Nph_shell
        

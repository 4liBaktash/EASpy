import numpy as np

from .auxiliary import eig_2d

class ImageParams:
    """
    Simple two level cleaning and calculating Hillas parameters
    """
    
    def __init__(self, _image_raw, pixel_pos, tailcuts):
        self._image_cleaned = None
        self._lvl1 = tailcuts[0]
        self._lvl2 = tailcuts[1]
        self._pixel_pos = pixel_pos
        
        self.pipe(_image_raw)
    
    
    def pipe(self, image_raw):
        self.tailcuts_cleaner(image_raw)
        self.calc_HillasParams
        return
    
    
    def tailcuts_cleaner(self, image_raw):
        lvl1 = np.where(image_raw > self._lvl1)
        self._image_cleaned = np.zeros_like(image_raw)
        self._image_cleaned[lvl1] = image_raw[lvl1]
        
        return
    
    
    @property
    def cleaned_image_pix(self):
        return np.where(self._image_cleaned > 0.)
    
    
    @property
    def calc_HillasParams(self):
        """
        This code was kindly provided by Micheal Blanc.
        Basic Hillas analysis.
        """
        ny = self._pixel_pos[0] #np.tile(self._pixel_pos, len(self._pixel_pos))
        nx = self._pixel_pos[1] #np.repeat(self._pixel_pos, len(self._pixel_pos))
        na = self._image_cleaned #self._image_cleaned.ravel()
    
        xa = nx * na
        ya = ny * na
        
        m10 = np.sum(xa, axis=-1)
        m01 = np.sum(ya, axis=-1)
        m00 = np.sum(na, axis=-1)
        m20 = np.sum(nx * xa, axis=-1)
        m02 = np.sum(ny * ya, axis=-1)
        m11 = np.sum(nx * ya, axis=-1)
        m30 = np.sum(nx * nx * xa, axis=-1)
        m03 = np.sum(ny * ny * ya, axis=-1)
        m21 = np.sum(ny * nx * xa, axis=-1)
        m12 = np.sum(nx * ny * ya, axis=-1)

        cog_x = m10/m00
        cog_y = m01/m00
        
        mu20 = m20 - m10 * cog_x  # m20 -<x>m10
        mu02 = m02 - m01 * cog_y  # m02 -<y>M01
        mu11 = m11 - m01 * cog_x  # mu11 = m11 - <x> M01

        # covariance matrix is obtained by division by
        # M00 (=mu00), cov=((mu20_prime,mu11_prime),(mu11_prime,mu02_prime)):
        # print "M00 = ", M00
        mu20_prime = mu20 / m00
        mu02_prime = mu02 / m00
        mu11_prime = mu11 / m00

        covM = np.stack((np.stack((mu20_prime, mu11_prime), axis=-1),
                         np.stack((mu11_prime, mu02_prime), axis=-1)), 
                         axis=-1)
        
        dist_cog_center = np.hypot(cog_x, cog_y)
        
        # eigenvectors of covM are major and minor axis:
        eig_w, eig_v = eig_2d(covM)

        phi = 0.5 * np.arctan2(2.0 * mu11_prime, mu20_prime - mu02_prime)
        
        beta = np.arctan2(cog_y, cog_x)  # the angle between x-axis and COG vector
        a = np.rad2deg(beta - phi)
        a = np.where(a > 90., a - 180., a)  # peak around 180deg
        a = np.where(a < -90, a + 180., a)
        center_alpha = np.fabs(np.deg2rad(a))
        
        # eigenvalues of covariance matrix:
        term1 = (mu20_prime + mu02_prime) / 2.
        term2 = np.sqrt(np.power((mu20_prime + mu02_prime) / 2, 2)\
                 - (mu02_prime * mu20_prime - mu11_prime**2))
        lambda1 = term1 + term2
        lambda2 = term1 - term2

        length = np.sqrt(np.abs(lambda1))
        width = np.sqrt(np.abs(lambda2))
        
        self.size = m00
        self.cog_x = cog_x
        self.cog_y = cog_y
        self.dist_cog_center = dist_cog_center
        self.width = width
        self.length = length
        self.phi = phi
        self.npix = len(self.cleaned_image_pix[0])

        print(self)
        return 
    
    
    @property
    def params2dict(self):
        resu = {}
        keys = ["size", "cog_x", "cog_y", "dist_cog_center", "width", "length", 
                "phi", "npix"]
        
        for key in keys:
            resu[key] = getattr(self, key)
            
        return resu
    
    
    def __str__(self):
        return f"size = {self.size:.4f},\ncog_x = {self.cog_x:.4f},\ncog_y = {self.cog_y:.4f},\n"\
               f"distance_cog_center = {self.dist_cog_center:.4f},\nwidth = {self.width:.4f},\n"\
               f"length = {self.length:.4f},\nphi = {self.phi:.4f},\nnpix = {self.npix}"

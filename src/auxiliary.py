import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon, Ellipse, Circle, Rectangle
from matplotlib.collections import PatchCollection
from numba import njit, prange

from .ctypes_wrapper import c_norm
from .multi_threading_params import _NUM_THREADS

def Point2Vector(point1, point2):
    """
    Construct vector from two points. 
    Vector will be pointing towards point2.
    """
    
    return point2 - point1


def Vec2Unit(v):
    "Calculate unit vector"
    
    if(len(v.shape) == 1):
        return v/np.linalg.norm(v)
    else:
        return v/np.linalg.norm(v, axis = -1)[..., np.newaxis]
            

def Perpendicular_Vector(v):
    """ Finds an arbitrary perpendicular vector to v."""
    
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array([1, 0, 0])
    if v[1] == 0:
        return np.array([0, 1, 0])
    if v[2] == 0:
        return np.array([0, 0, 1])

    # arbitrarily set a = b = 1 
    #(one equation for 3 unknowns -> 2 degrees of freedom)
    # then the equation simplifies to
    #     c = -(x + y)/z
    return np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]])


def Orthonormal_Vectors(v):
    """
    Calculate two vectors such that v,v1,v2 form an orthonormal set of vectors.
    """
    
    V1 = Perpendicular_Vector(v)
    unit_V1 = Vec2Unit(V1)
    V2 = np.cross(v, unit_V1)
    unit_V2 = Vec2Unit(V2)
    
    return unit_V1, unit_V2


def calc_rel_dist(tel_pos, coords):
    """
    Calculate relative distance of voxels to telescope position
    """

    return c_norm(coords, tel_pos, _NUM_THREADS)


def neighbours(x,y):
    """
    Returns neighbour pixel ids in a rectangular grid
    """
    
    resu = [[x, y-1], [x, y+1], 
            [x-1, y-1], [x-1, y], [x-1, y+1], 
            [x+1, y-1], [x+1, y], [x+1, y+1]]
    
    return resu


def eig_2d(matrix):
    """ 
        This code was kindly provided by Micheal Blank.
        calculate the eigenvalues and eigenvectors of 2d matrices.
        Returns (eigenvalues, eigenvectors)
    """
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]
    T = (a + d)/2
    D = a * d - b * c
    C = T**2 - D
   # sel0 = C >= 0
   # C[np.bitwise_not(sel0)] = np.nan
    C = np.sqrt(C)  # Can also be complex
    
    L1 = T + C
    L2 = T - C
    sel1 = c != 0
    sel2 = b != 0
    sel3 = np.bitwise_not(np.bitwise_and(sel1, sel2))
    ev = np.zeros_like(matrix)
    sax = -1
    
    if sel3.any():
        ev[sel3] = np.array([[1, 0], [0, 1]])
        
    if sel1.any():
        n1 = np.sqrt((L1-d)**2 + c**2)
        n2 = np.sqrt((L2-d)**2 + c**2)
        ev[sel1] = np.stack([np.stack([(L1-d)/n1, c/n1], axis=sax),
                             np.stack([(L2-d)/n2, c/n2], axis=sax)],
                            axis=sax)[sel1]
        
    if sel2.any():
        n1 = np.sqrt((L1-a)**2 + b**2)
        n2 = np.sqrt((L2-a)**2 + b**2)
        ev[sel2] = np.stack([np.stack([b/n1, (L1-a)/n1], axis=sax),
                             np.stack([b/n2, (L2-a)/n2], axis=sax)],
                            axis=sax)[sel2]
        
    return np.array([L1, L2]).T, ev


def plot_ellipse(cog_x, cog_y, length, width, phi,
                 color='g', alpha=0.6, ax=None):
    
    if ax is None:
        ax = plt.gca()

    ellipses = []
    for i in [2, 4]:
        ellipses.append(Ellipse((cog_x, cog_y),
                                i * length,
                                i * width,
                                phi * 180./np.pi,
                                ))
    ellipse_collection = PatchCollection(ellipses)
    ellipse_collection.set_edgecolors(3 * [color])
    ellipse_collection.set_alpha(alpha)
    ellipse_collection.set_facecolor('none')
    ellipse_collection.set_linewidth(2)
    ax.add_collection(ellipse_collection)

    return ellipse_collection


def draw_patches(x_coords, y_coords, pixel_shape, sizehex=0.03, orient=0.):
    patches = []
    if(pixel_shape == 0):
        for x,y in zip(x_coords, y_coords):
            patches.append(Rectangle(xy = (x,y), width = sizehex, height = sizehex))
    elif(pixel_shape == 1):
        for x,y in zip(x_coords, y_coords):
            patches.append(Circle(xy = (x,y), radius = sizehex/2.))
    elif(pixel_shape == 2):
        for x, y in zip(x_coords, y_coords):
            patches.append(
                RegularPolygon(
                    xy = (x, y),
                    numVertices = 6,
                    # sizehex is the flat to flat width (diameter of the incircle) of the hexagonal pixel
                    # Radius of circumscribed circle r is related to the diameter of the incircle i via 
                    #r = i / sqrt(3)
                    radius = sizehex/np.sqrt(3),
                    orientation = orient,  # in radians
                )
            )
    else:
        raise ValueError("Pixel shape not supported!")
    return patches


def plot_iact_image(image,
                    x_coords,
                    y_coords,
                    pixel_shape,
                    ax=None,
                    vmin=None,
                    vmax=None,
                    edgecolor='k',
                    cmap='cividis',
                    lim=0.41,
                    sizehex=0.03,
                    orient=0.):
    """
    This code was kindly provided by Micheal Blank.
    Original code snippet is from https://github.com/fact-project/pyfact
    """
    # Based on the camera implementation in pyfact
    if ax is None:
        ax = plt.gca()

    ax.set_aspect('equal')

    # if the axes limit is still (0,1) assume new axes
    if ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    #ax.set_xlim(0., lim)
    #ax.set_ylim(-lim, lim)
    

    if vmin is None:
        vmin = np.min(image)
    if vmax is None:
        vmax = np.max(image)

    edgecolors = np.array(len(image) * [edgecolor])

    patches = draw_patches(x_coords, y_coords, pixel_shape, sizehex, orient)
    collection = PatchCollection(patches, picker=0)
    collection.set_edgecolors(edgecolors)
    collection.set_cmap(cmap)
    collection.set_array(image)
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)

    plt.draw_if_interactive()

    return collection


@njit(parallel=True)
def numba_apply_qeff_abs(Nph, height_idx, LUT, wvl_mask, scaling, 
                         denom, eta, emission, wvl = None):
    
    resu = np.zeros_like(Nph)
    
    for i in prange(resu.size):   
    
        tmp  = LUT[wvl_mask, height_idx[i]]
        #!scaling only needed for plane parallel atmosphere!
        tmp /= scaling[i]
        tmp *= -1
        
        tmp = np.exp(tmp)
        tmp *= eta        
        
        if(emission=="cherenkov"):
            expected_absorb  = np.trapz(tmp, wvl)
        #no continous emission for fluorescene     
        else:
            expected_absorb  = np.sum(tmp)
        expected_absorb /= denom

        resu[i]  = Nph[i] 
        resu[i] *= expected_absorb
        
    return resu

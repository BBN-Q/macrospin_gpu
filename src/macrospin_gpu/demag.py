from __future__ import division

import scipy as sp
import numpy as np
import scipy.special as special
import scipy.integrate as integrate

def demagCylinder(length, width, height, cgs=False):
    '''Returns the demag factors of an elliptical nanopillar'''
    a = 0.5*length # Semimajor
    b = 0.5*width  # Semiminor
    d = 0.5*height # Semi...thickness?

    beta = b/a
    eps  = np.sqrt(1.0 - beta**2)
    
    xi = d/b

    g  = lambda phi: np.sqrt(1.0 - (eps*np.cos(phi))**2)
    h  = lambda phi: (np.cos(phi)/g(phi))**2
    i1 = lambda phi: h(phi)*special.hyp2f1(-0.5,0.5,2.0,-1.0/(xi*xi*g(phi)**2))
    i2 = lambda phi: special.hyp2f1(-0.5,0.5,2.0,-1.0/(xi*xi*g(phi)**2))
    
    E = special.ellipe(eps**2) # complete elliptic int of second kind
    K = special.ellipk(eps**2) # complete elliptic int of second kind

    c1 = 8.0 / (3.0 * np.pi**2 * xi)
    c2 = 2.0 / (np.pi)

    Nxx = (c1/eps**2)*(beta*beta*K - E) 
    Nxx += c2*beta*beta*integrate.quad(i1, 0.0, 0.5*np.pi)[0]
    
    Nzz = 1.0 + c1*K
    Nzz += -c2*integrate.quad(i2, 0.0, 0.5*np.pi)[0]

    if cgs:
        answer = 4.0*np.pi*np.array([Nxx, 1.0-Nxx-Nzz, Nzz])
    else:
        answer = np.array([Nxx, 1.0-Nxx-Nzz, Nzz])
    return answer
    
def demagEllipsoid(length, width, height, cgs=False):
    '''Returns the demag factors of an ellipsoid'''
    a = 0.5*length # Semimajor
    b = 0.5*width  # Semiminor
    c = 0.5*height # Semi-more-minor

    b1 = b/a
    b2 = np.sqrt(1.0 - b1*b1)
    
    c1 = c/a
    c2 = np.sqrt(1.0 - c1*c1)

    d2 = b2/c2
    d1 = np.sqrt(1.0 - d2*d2)

    theta = np.arccos(c1)
    m = d2*d2

    E = special.ellipeinc(theta, m) # incomplete elliptic int of second kind
    F = special.ellipkinc(theta, m) # incomplete elliptic int of first kind

    Nxx = (b1*c1)/(c2**3 * d2**2) * (F - E)
    Nyy = (b1*c1)/(c2**3 * d2**2 * d1**2) * (E - d1*d1*F - d2*d2*c2*c1/b1)
    Nzz = (b1*c1)/(c2**3 * d1**2) * (c2*b1/c1 - E)

    if cgs:
        return np.pi*4.0*np.array([Nxx, Nyy, Nzz])
    else:
        return np.array([Nxx, Nyy, Nzz])


if __name__=='__main__':

    # Some example use cases
    a = demagCylinder(100,50,2)
    b = demagCylinder(100,50,0.75)
    c = demagCylinder(80,45,2)

    # These are the IP anisotropies
    Ms = 1200.0
    print(a)
    print("Nyy-Nxx:", 4.0*np.pi*Ms*(a[1]-a[0]), "Oe")
    print("Nyy-Nxx:", 4.0*np.pi*Ms*(a[1]-a[0])*(1e3/(4.0*np.pi)), "A/m")
    print("Nzz:", a[2],  4.0*np.pi*Ms*(a[2]), "Oe")
    print("Nzz:", a[2],  4.0*np.pi*Ms*(a[2])*(1e3/(4.0*np.pi)), "A/m")
    print(4.0*np.pi*Ms*(b[1]-b[0]))

    Ms = 640.0
    print(4.0*np.pi*Ms*(c[1]-c[0]))

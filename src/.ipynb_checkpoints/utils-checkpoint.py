import numpy as np
from itertools import permutations
from functools import reduce
import scipy.special
import multiprocessing
import scipy.optimize


def multiprocess_sim(assays, nproc=None):
    """
    Run the method `assay.run()` in parallel processes for each assay
    in `assays` list.
    This method returns the list of `assay.run()` outputs for every assay.
    """
    
    if (nproc == None):
        nproc = len(assays)

    pool = multiprocessing.Pool(min(12, nproc))
    
    out = []
    for assay in assays:
        out.append(pool.apply_async(assay.run))
        
    pool.close()
    pool.join()
    
    return [res.get() for res in out]


    
def skewed_gaussian(x, sk):
    """
    A skewed gaussian to initialize ntot
    """
    return np.exp(-x**2/2)*(1 +scipy.special.erf(sk*x/np.sqrt(2)))


def vtau_lin(p):
    return p.r0*((p.beta/(p.alpha + p.gamma))**(1/p.M) - 1)**(-1)


def sel_coef_lin(p, vt):
    return (p.alpha + p.gamma)*p.M/vt


def vtau_lin_virulence(p):
    return p.r0*((p.beta*np.sqrt(p.alpha)/(p.alpha + p.gamma))**(1/p.M) - 1)**(-1)


def speed_FKPP(p, size, cutoff):
    return 2*np.sqrt((p.beta-p.alpha-p.gamma)*p.D_coef)*(1 - np.pi**2/np.log(size/cutoff)**2)


def speed_lin(p, size, cutoff):
    vt = vtau_lin(p)
    s = sel_coef_lin(p, vt)
    d_inv = (s/p.D_coef)**(1/3)
    eps0 = - 2.3381
    r = size/cutoff
    aux = 3*np.log(r*d_inv)
    return 2*p.D_coef*d_inv*(aux**(1/3) + eps0*aux**(-1/3))

def size_lin(p, cutoff):
    
    vtau = vtau_lin(p)
    def _Neq(x, p, cutoff, vtau):
        return (vtau - speed_lin(p, x, cutoff)*p.M*p.Nh/x)

    return scipy.optimize.brentq(_Neq, 1e4, p.Nh*1000, args = (p, cutoff, vtau))


def _get_neighs(i, net, boundary = False, clear = False):
    
    def _try_ravel(idxs, net):
            
        try:
            ind = np.ravel_multi_index(idxs.astype(int), net)
            
        except ValueError:
            ind = -1
                
        return ind
        
    idxs = np.asarray(np.unravel_index(i,  net))
    dim = len(net)
        
    ia = np.zeros(dim)
    ia[0] = 1
    
    uset = set(permutations(ia)) | set(permutations(-ia))
    indx = []
    for neigh in uset:
        ids = np.array(neigh) + idxs
        if(boundary):
            for i, idx in enumerate(ids):
                if(ids[i] // (net[i]) > 0):
                    ids[i] = ids[i] % (net[i])
                if(ids[i] < 0):
                    ids[i] = ids[i] % (net[i])
        
        indx.append(ids)
        
    ind = np.array([_try_ravel(ids, net) for ids in indx])
        
    if(clear == True):
        ind = ind[ind != -1]
            
    return ind


def init_neighs(net, boundary = False):
    

    size = reduce((lambda x,y: x*y), net)
    z = 2*len(net)
                  
    neighs = np.empty((size, z), dtype = int)

    for i in range(size):
        neighs[i,:] = _get_neighs(i, net, boundary = boundary)
        

    aneighs = neighs
    
    return aneighs


def dt_from_cfl(cfl, D, dx):
    return cfl * dx**2 / D

def lin_from_two_points(x, x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)*(x-x1) + y1
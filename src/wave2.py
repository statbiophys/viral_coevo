import wave as w
import numpy as np
import numba_functions as nbf
import utils as ut


class Vwave2_pars(w.Vwave_pars):

    def __init__(self, assay_id, tot_time, dx, r0, M, D_coef, D_coef_m, beta, beta_m, gamma, gamma_m,
                 alpha, alpha_m, eps, is_flux=False, t_burn_cutoff=1500, t_burn=2000, cutoff=1, back_width_fract=5,
                 n_x_bins=2**12, N0=1e8, Nh=1e10, dt=0.05, traj_step=100, check_step=1000, traj_after_burn=False,
                 verbose=True):
        """
        Parameters for two competing waves. Parameters with _m refer to the mutant population that appears
        after t_burn. If flux=False the mutant initial disribution is inserted after t_burn and if 
        follows the resident multiplied by eps.
        If flux=True after t_burn a new flux of mutant starts with rate eps.
        """
        w.Vwave_pars.__init__(self, assay_id, tot_time, dx, r0, M, D_coef, alpha, beta,
                             gamma, t_burn_cutoff, t_burn, cutoff, back_width_fract,
                             n_x_bins, N0, Nh, dt, traj_step, check_step, traj_after_burn, verbose)
        
        self.D_coef_m = D_coef_m
        self.beta_m = beta_m
        self.gamma_m = gamma_m
        self.alpha_m = alpha_m
        self.is_flux = is_flux
        self.eps = eps
        
        
class Traj_Vwave2(w.Traj):

    def start(self, wave):
        
        self.n_mut, self.n_res = np.array([]), np.array([])
        w.Traj.start(self, wave)
    
    
    def update(self, time):
        
        self.n_mut = np.append(self.n_mut, self.wave.n2[1,:].sum() * self.wave.p.dx)
        self.n_res = np.append(self.n_res, self.wave.n2[0,:].sum() * self.wave.p.dx)
        w.Traj.update(self, time)
        
        
class Vwave2(w.Vwave) :
    
    def __init__(self, pars, traj=Traj_Vwave2(), init_n=None, init_nh=None):
        w.Vwave.__init__(self, pars, traj, init_n, init_nh)
        
        
    def _start(self):
        self.m_init = False
        w.Vwave._start(self)
    
        
    def _init_n(self, B, x0, xvals):
        
        n2 = np.zeros((2, self.p.n_x_bins))
        x_range = ((xvals - x0)*self.p.dx - 0.2*self.p.r0) / (0.6*self.p.r0)
        n2[0,:] = ut.skewed_gaussian(x_range, 4)
        n2[0,:] = self.p.N0*n2[0,:]/(n2[0,:].sum()*self.p.dx)
        
        # number of residents and mutants
        self.n2 = n2.astype(np.float64)
        # Marginal over x
        self.n = n2.sum(axis=0)
    
        
    def _init_n_by_prior(self):
        self.n2 = self.init_n.astype(np.float64)
        return self.n2.sum(axis=0)
        
        
    def _step(self, t):
        
        if t*self.p.dt > self.p.t_burn and not self.m_init:
            self.m_init = True
            if not self.p.is_flux and type(self.init_n) != np.ndarray:
                self.n2[1,:] = self.p.eps * self.n2[0,:] 
                self.n2[0,:] -= self.n2[1,:] 
        w.Vwave._step(self, t)
        
        
    def _update_n(self):
        
        p = self.p
        self.n2 = nbf.compute_n2(self.n2, self.P, self.p.is_flux, self.m_init, self.p.eps, p.D_coef, p.D_coef_m, p.dt, p.dx, p.beta, 
                                 p.beta_m, p.gamma, p.gamma_m, p.alpha, p.alpha_m, p.M, self.cutoff, p.back_width_fract, 
                                 self.back_i, self.max_i, self.front_i)
        self.n = self.n2.sum(axis=0)
import wave as w
import numpy as np
import numba_functions as nbf
import utils as ut


class Vwave_virulence_pars(w.Vwave_pars) :
    
    def __init__(self, assay_id, tot_time, dx, r0, M, D_coef, n_alpha_bins, eps, beta, gamma, 
                 alphas=None, alpha0_i=None, t_burn_cutoff=1500, t_burn=2000, cutoff=1, global_cutoff=False, back_width_fract=2000,
                 n_x_bins=2**12, N0=1e8, Nh=1e10, dt=0.05, traj_step=100, check_step=1000, traj_after_burn=False, verbose=True):
        
        w.Vwave_pars.__init__(self, assay_id, tot_time, dx, r0, M, D_coef, alphas, beta,
                 gamma, t_burn_cutoff, t_burn, cutoff, back_width_fract,
                 n_x_bins, N0, Nh, dt, traj_step, check_step, traj_after_burn, verbose)
        
        self.n_alpha_bins = n_alpha_bins
        self.eps = eps
        self.global_cutoff = global_cutoff

        if type(alphas) != list and type(alphas) != np.ndarray:
            self.alphas = np.linspace(0.15, 1.5, n_alpha_bins)
        else:
            self.alphas = alphas
            n_alpha_bins = len(alphas)
            
        if alpha0_i == None:
            self.alpha0_i = int(n_alpha_bins/2)
        else:
            self.alpha0_i = alpha0_i
        
        self.dalpha = np.mean(np.diff(self.alphas))    
        self.alpha = self.alphas[self.alpha0_i] # It will be only used by Vwave to init vtau
        self.alpha_neighs = ut.init_neighs((self.n_alpha_bins,), boundary=False).astype(np.int64)

        
class Traj_virulence(w.Traj):

    def start(self, wave):
        
        self.mean_alpha, self.std_alpha = np.array([]), np.array([])
        self.max_alpha = np.array([])
        w.Traj.start(self, wave)
    
    
    def update(self, time):
        
        p = self.wave.p
        alpha_ab = self.wave.n_alpha.sum(axis=1)
        a = np.sum(alpha_ab*p.alphas)/alpha_ab.sum()
        self.mean_alpha = np.append(self.mean_alpha, a)
        max_a = self.wave.p.alphas[np.argmax(alpha_ab)]
        self.max_alpha = np.append(self.max_alpha, max_a)
        
        a2 = np.sum(alpha_ab*p.alphas**2)/alpha_ab.sum()
        stda = np.where(a2>a**2, np.sqrt(a2 - a**2, where=a2>a**2), 0)
        self.std_alpha = np.append(self.std_alpha, stda)
        
        w.Traj.update(self, time)
        b = p.beta * np.sqrt(a)
        self.f_tip[-1] = b * self.P_tip[-1] ** p.M - a - p.gamma
        self.s_tip[-1] = b * p.M * self.dP_tip[-1] * self.P_tip[-1]**(p.M-1)
        
        
class Vwave_virulence(w.Vwave) :
    
    def __init__(self, pars, is_stoch=False, traj=Traj_virulence()):
        w.Vwave.__init__(self, pars, traj)
        self.is_stoch = is_stoch
        
        
    def _start(self):
        self.eps = 0
        w.Vwave._start(self)
        
        
    def _init_n(self, B, x0, xvals):
        
        n_alpha = np.zeros((self.p.n_alpha_bins, self.p.n_x_bins))
        x_range = ((xvals - x0)*self.p.dx - 0.2*self.p.r0) / (0.6*self.p.r0)
        n_alpha[self.p.alpha0_i,:] = ut.skewed_gaussian(x_range, 4)
        n_alpha[self.p.alpha0_i,:] = self.p.N0*n_alpha[self.p.alpha0_i,:]/(n_alpha[self.p.alpha0_i,:].sum()*self.p.dx)
        
        # Matrix over x and alpha
        self.n_alpha = n_alpha.astype(np.float64)
        # Marginal over x
        self.n = n_alpha.sum(axis=0)*self.p.dalpha
        
        
    def _step(self, t):
        
        if t*self.p.dt > self.p.t_burn:
            if type(self.p.eps) == float:
                self.eps = self.p.eps
            else:
                # epsilon can be scheduled in time
                self.eps = self.p.eps.get(t*self.p.dt - self.p.t_burn)
        w.Vwave._step(self, t)
        
        
    def _update_n(self):
        
        p = self.p
        if self.is_stoch:
            self.n_alpha = nbf.compute_n_alpha_stoch(self.n_alpha, self.n, self.P, p.alpha_neighs, p.D_coef, self.eps, p.dt, p.dx, 
                                       p.beta, p.gamma, p.alphas, p.dalpha, p.M, self.cutoff, p.back_width_fract, 
                                       self.back_i, self.max_i, self.front_i, self.p.global_cutoff)
        else:
            self.n_alpha = nbf.compute_n_alpha(self.n_alpha, self.n, self.P, p.alpha_neighs, p.D_coef, self.eps, p.dt, p.dx, 
                                       p.beta, p.gamma, p.alphas, p.dalpha, p.M, self.cutoff, p.back_width_fract, 
                                       self.back_i, self.max_i, self.front_i, self.p.global_cutoff)
            
        self.n = self.n_alpha.sum(axis=0)*p.dalpha
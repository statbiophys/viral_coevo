import wave as w
import numpy as np
import numba_functions as nbf
import utils as ut


class Vwave_mut_rate_pars(w.Vwave_pars) :
    
    def __init__(self, assay_id, tot_time, dx, r0, M, eps, alpha, beta, gamma, lambda_tilde,
                 Ds, D0_i=None, t_burn_cutoff=1500, t_burn=2000, cutoff=1, global_cutoff=False, back_width_fract=5,
                 n_x_bins=2**12, N0=1e8, Nh=1e10, dt=0.05, traj_step=100, check_step=1000, traj_after_burn=False, verbose=True):
        
        w.Vwave_pars.__init__(self, assay_id, tot_time, dx, r0, M, 0, alpha, beta,
                 gamma, t_burn_cutoff, t_burn, cutoff, back_width_fract,
                 n_x_bins, N0, Nh, dt, traj_step, check_step, traj_after_burn, verbose)
        
        # lambda tilde is the factor multiplying D in the mutational load
        # lambda = d to b ratio = lambda_tilde <dx^2> / 2
        self.lambda_tilde = lambda_tilde
        
        self.n_D_bins = len(Ds)
        self.eps = eps
        self.global_cutoff = global_cutoff
        
        self.Ds = Ds
        if self.n_D_bins == 1: 
            self.dD = 1
        else:
            self.dD = np.mean(np.diff(self.Ds))
            
        if D0_i == None:
            self.D0_i = int(self.n_D_bins/2)
        else:
            self.D0_i = D0_i
              
        self.D_coef = self.Ds[self.D0_i]
        self.D_neighs = ut.init_neighs((self.n_D_bins,), boundary=False).astype(np.int64)
        
        
class Traj_mut_rate(w.Traj):

    def start(self, wave):
        
        self.mean_D, self.std_D = np.array([]), np.array([])
        self.max_D = np.array([])
        w.Traj.start(self, wave)
    
    
    def update(self, time):
        
        p = self.wave.p
        D_ab = self.wave.n_D.sum(axis=1)
        D_ab_norm = D_ab.sum()
        D = np.sum(D_ab*p.Ds)/D_ab_norm
        self.mean_D = np.append(self.mean_D, D)
        max_D = self.wave.p.Ds[np.argmax(D_ab)]
        self.max_D = np.append(self.max_D, max_D)
        
        D2 = np.sum(D_ab*p.Ds**2)/D_ab_norm
        stdD = np.where(D2>D**2, np.sqrt(D2 - D**2, where=D2>D**2), 0)
        self.std_D = np.append(self.std_D, stdD)
        
        w.Traj.update(self, time)
        #i_tip = round(self.x_tip[-1] / p.dx)
        self.f_tip[-1] = p.beta * self.P_tip[-1] ** p.M * (1 - p.lambda_tilde * D) - p.alpha - p.gamma 
        self.s_tip[-1] = p.beta * p.M * self.dP_tip[-1] * self.P_tip[-1]**(p.M-1) * (1 - p.lambda_tilde * D)
        
        
class Vwave_mut_rate(w.Vwave) :
    
    def __init__(self, pars, is_stoch=False, traj=Traj_mut_rate(), init_n=None, init_nh=None):
        w.Vwave.__init__(self, pars, traj, init_n, init_nh)
        self.is_stoch = is_stoch
        
        
    def _start(self):
        self.eps = 0.0
        w.Vwave._start(self)
        
    def _init_n(self, B, x0, xvals):
        
        n_D = np.zeros((self.p.n_D_bins, self.p.n_x_bins))
        x_range = ((xvals - x0)*self.p.dx - 0.2*self.p.r0) / (0.6*self.p.r0)
        n_D[self.p.D0_i,:] = ut.skewed_gaussian(x_range, 4)
        n_D[self.p.D0_i,:] = self.p.N0*n_D[self.p.D0_i,:]/(n_D[self.p.D0_i,:].sum()*self.p.dx)
        
        # Matrix over x and D
        self.n_D = n_D.astype(np.float64)
        # Marginal over x
        self.n = n_D.sum(axis=0)*self.p.dD
    
    
    def _init_n_by_prior(self):
        if self.p.n_x_bins != len(self.init_n):
            raise Exception('Invalid number of bins of the initial condition')
            
        n_D = np.zeros((self.p.n_D_bins, self.p.n_x_bins))
        n_D[self.p.D0_i,:] = self.init_n.astype(np.float64)/self.p.dD
        self.n_D = n_D.astype(np.float64)
        return self.init_n
        
        
    def _step(self, t):
        
        if t*self.p.dt > self.p.t_burn:
            if type(self.p.eps) == float:
                self.eps = self.p.eps
            else:
                # epsilon can be scheduled in time
                self.eps = self.p.eps.get(t*self.p.dt - self.p.t_burn)
        
            #print(self.eps, t*self.p.dt - self.p.t_burn, self.p.eps(t*self.p.dt - self.p.t_burn))
        
        w.Vwave._step(self, t)
        
        
    def _update_n(self):
        
        p = self.p
        if self.is_stoch:
            self.n_D = nbf.compute_n_D_stoch(self.n_D, self.n, self.P, p.D_neighs, self.eps, p.dt, p.dx, p.alpha,
                                       p.beta, p.gamma, p.lambda_tilde, p.Ds, p.dD, p.M, self.cutoff, p.back_width_fract, 
                                       self.back_i, self.max_i, self.front_i, self.p.global_cutoff)
        else:
            self.n_D = nbf.compute_n_D(self.n_D, self.n, self.P, p.D_neighs, self.eps, p.dt, p.dx, p.alpha,
                                       p.beta, p.gamma, p.lambda_tilde, p.Ds, p.dD, p.M, self.cutoff, p.back_width_fract, 
                                       self.back_i, self.max_i, self.front_i, self.p.global_cutoff)
        
        self.n = self.n_D.sum(axis=0)*p.dD
            
import wave as w
import numpy as np
import numba_functions as nbf
import utils as ut


class Newton_pars:
    
    def __init__(self, learn_rate, t_newton, t_newton_step, t_speed=100, final_av_t_fract=0.1):
        # Learning rate of the update
        self.lr = learn_rate
        # Total time of the process
        self.t = t_newton
        # Time in between updates
        self.t_step = t_newton_step
        # Time over which the wave speed is averaged
        self.t_speed = t_speed
        # Fraction of the total time over which the final D_coef is averaged
        self.final_av_t_fract = final_av_t_fract
        

class Vwave_newton_mut_rate_pars(w.Vwave_pars) :
    
    def __init__(self, assay_id, dx, dt, t_burn_cutoff, t_burn, newton_pars,
                 r0, M, R0, recovery, subs_rate, Ud, Ub_guess, Nh, N0, dx2,
                 cutoff=1, global_cutoff=True, back_width_fract=5,
                 n_x_bins=2**12, traj_step=100, check_step=1000, traj_after_burn=False, verbose=True):
        
        # Antigenic mutational step^2
        self.dx2 = dx2
        # reproductive ratio
        self.R0 = R0
        # recovery rate = alpha + gamma
        self.recovery = recovery
        alpha=0
        # substitution rate
        self.subs = subs_rate
        
        # deleterious mutation rate
        self.Ud = Ud
        # Initial guess for the beneficial mut rate, it will be corrected 
        # by the relation v/dx2 = ns during the burning time
        self.Ub_guess = Ub_guess
        self.D_coef = Ub_guess * dx2 / 2
        
        # the transmissibility is obtained from R0
        # It already contains the mutational load.
        beta = R0 * recovery
        
        # The trajectory has to be computed during burining time to 
        # have the speed to update D_coef
        traj_after_burn = False
        
        self.nwt = newton_pars
        
        w.Vwave_pars.__init__(self, assay_id, newton_pars.t, dx, r0, M, self.D_coef, alpha, beta,
                 recovery, t_burn_cutoff, t_burn, cutoff, back_width_fract,
                 n_x_bins, N0, Nh, dt, traj_step, check_step, traj_after_burn, verbose)
        
        
class Traj_mut_rate_data(w.Traj):

    def start(self, wave):
        self.D = np.array([])
        w.Traj.start(self, wave)
    
    def update(self, time):
        p = self.wave.p
        self.D = np.append(self.D, self.wave.p.D_coef)
        w.Traj.update(self, time)
        
        
class Vwave_newton_mut_rate(w.Vwave) :
    
    def __init__(self, pars, traj=Traj_mut_rate_data(), init_n=None, init_nh=None):
        w.Vwave.__init__(self, pars, traj, init_n, init_nh)        
        
        
    def _start(self):
        
        self.eps = 0
        w.Vwave._start(self)
        
        
    def _init_n(self, B, x0, xvals):
        
        n = np.zeros(self.p.n_x_bins)
        x_range = ((xvals - x0)*self.p.dx - 0.2*self.p.r0) / (0.6*self.p.r0)
        n = ut.skewed_gaussian(x_range, 4)
        n = self.p.N0*n/(n.sum()*self.p.dx)
        self.n = n.astype(np.float64)
        
        # Length of the antigenic space
        self.L = self.p.n_x_bins * self.p.dx
        self.speed_delay_i = int(np.floor(self.p.nwt.t_speed/(self.p.traj_step*self.p.dt)))
        
        
    def _init_n_by_prior(self):
        if self.p.n_x_bins != len(self.init_n):
            raise Exception('Invalid number of bins of the initial condition')
        
        # Length of the antigenic space
        self.L = self.p.n_x_bins * self.p.dx
        self.speed_delay_i = int(np.floor(self.p.nwt.t_speed/(self.p.traj_step*self.p.dt)))
        return self.init_n.astype(np.float64) 
    
    
    def _step(self, t):
        
        p = self.p
        if t*p.dt > p.t_burn and t*p.dt > self.p.nwt.t_speed and t%p.nwt.t_step == 0:
            self._newton_update_Ub()
            
        w.Vwave._step(self, t)
        
        if t == self.n_steps-1:
            i_av_steps = int(np.floor(p.nwt.final_av_t_fract*p.nwt.t/(p.traj_step*p.dt)))
            p.D_coef = np.mean(self.traj.D[-i_av_steps:])
            
        
    def _newton_update_Ub(self):
        
        dx = self.traj.x_front[-1]-self.traj.x_front[-self.speed_delay_i-1]
        dt = self.traj.times[-1]-self.traj.times[-self.speed_delay_i-1]
        # np.where deals with the boundary conditions
        speed = np.where(dx > 0, dx, dx+self.L)/dt
        
        # function whose zero has to be found
        f = speed/np.sqrt(self.p.dx2) - self.p.subs
        
        # Derivative of the function assuming a linear fitness regime
        # coefficients will be absorbed in the learning rate
        dspeed = self.p.D_coef**(2/3) / np.sqrt(self.p.dx2)
        
        self.D_new = self.p.D_coef - self.p.nwt.lr * f / dspeed
        self.p.D_coef = self.D_new
            
        
    def _update_n(self):
        
        p = self.p
        self.n = nbf.compute_new_n(self.n, self.P, p.D_coef, p.dt, p.dx, p.alpha,
                                   p.beta, p.gamma, p.M, self.cutoff, p.back_width_fract, 
                                   self.back_i, self.max_i, self.front_i)

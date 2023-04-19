from tqdm.notebook import tqdm
import numpy as np
import utils as ut
import numba_functions as nbf


class Vwave_pars:
    """
    Container of readonly parameters for a run of the viral wave simulation
    """
    
    def __init__(self, assay_id, tot_time, dx, r0, M, D_coef, alpha, beta,
                 gamma, t_burn_cutoff=1500, t_burn=2000, cutoff=1, back_width_fract=5,
                 n_x_bins=2**12, N0=1e8, Nh=1e10, dt=0.05, 
                 traj_step=100, check_step=1000, traj_after_burn=False, verbose=True):
        
        # Assay identifier
        self.id = assay_id
        # Number of temporal steps
        self.tot_time = tot_time
        # Size of temporal step
        self.dt = dt
        # How many steps between trajectories update
        self.traj_step = traj_step
        # Whether to compute the traj only after the burning time
        self.traj_after_burn = traj_after_burn
        # How many steps between sanity checks of the wave
        self.check_step = check_step
        # Time to wait before the cutoff is applied
        self.t_burn_cutoff = t_burn_cutoff
        # Time to wait before considering the wave stationary
        self.t_burn = t_burn
        self.verbose = verbose
        
        if t_burn_cutoff > t_burn:
            raise Exception('Cutoff burning time larger than stationary burning time')
            
        # Value of cutoff when applied
        self.cutoff = cutoff
        # Fraction of the wave width (computed between front and max) to be updated in 
        # the back. Beyond this width the wave is killed
        self.back_width_fract = back_width_fract
        # Number of x bins (antigenic space)
        self.n_x_bins = n_x_bins
        # Resolution of the antigenic space
        self.dx = dx
        # Initial number of hosts
        self.N0 = N0
        # Cross reactivity of immune protection
        self.r0 = r0
        # Number of receptors per host
        self.M = M
        # Transmission coef
        self.beta = beta
        # Diffusion coefficient of the virus
        self.D_coef = D_coef
        # Total population of hosts
        self.Nh = Nh
        # Virulence of the virus
        self.alpha = alpha
        # Recovery rate
        self.gamma = gamma
        
        self._check_precision()
        
        
    def _check_precision(self):
        
        self.kernel_prec = self.dx/self.r0
        if self.kernel_prec > 0.1:
            print("proc.", self.id, 'WARNING: dx is not small compared to r0. dx/r0=', self.kernel_prec)
        
        self.CFL = self.D_coef*self.dt/self.dx**2
        if self.kernel_prec > 0.1:
            print("proc.", self.id, 'WARNING: CFL is not small, choose a smaller dt. CFL=', self.CFL)
            
            
            
class Traj:
    """
    Container of Viral wave information. 
    Updated along the wave evolution every traj_step
    """
    def start(self, wave):
        self.nvirus, self.times = np.array([]), np.array([])
        self.x_front, self.x_back = np.array([]), np.array([])
        self.x_tip, self.s_tip, self.f_tip = np.array([]), np.array([]), np.array([])
        self.P_tip, self.dP_tip = np.array([]), np.array([])
        self.wave = wave
    
    def update(self, time):
        self.times = np.append(self.times, time * self.wave.p.dt)
        self.nvirus = np.append(self.nvirus,self.wave.n.sum() * self.wave.p.dx)
        self.x_front = np.append(self.x_front, self.wave.front_i * self.wave.p.dx)
        self.x_back = np.append(self.x_back, self.wave.back_i * self.wave.p.dx)
        p = self.wave.p
        
        c = self.wave.cutoff
        if self.wave.is_stoch: c = 0.5
        xt_i = nbf.find_first_zero(self.wave.n, self.wave.max_i, c)
        dxt_i = (self.wave.n[xt_i-1]-c)/(self.wave.n[xt_i-1] - self.wave.n[xt_i])
        self.x_tip = np.append(self.x_tip, (xt_i + dxt_i) * self.wave.p.dx)
        Pt = self.wave.P[xt_i-1] + (self.wave.P[xt_i]-self.wave.P[xt_i-1])*dxt_i
        dPt = (self.wave.P[xt_i] - self.wave.P[xt_i-1]) / p.dx
        self.P_tip = np.append(self.P_tip, Pt)
        self.dP_tip = np.append(self.dP_tip, dPt)
        self.f_tip = np.append(self.f_tip, p.beta * Pt**p.M - p.alpha - p.gamma)
        self.s_tip = np.append(self.s_tip, p.beta * p.M * dPt * Pt**(p.M-1))
        
    def speed(self, delta=1):
        l = self.wave.p.n_x_bins * self.wave.p.dx
        d = self.x_front[delta:]-self.x_front[:-delta]
        return np.where(d > 0, d, d+l)/(self.times[delta:]-self.times[:-delta])
    
    def width(self):
        l = self.wave.p.n_x_bins * self.wave.p.dx
        d = self.x_front-self.x_back
        return np.where(d>0, d, d+l)
    
    
class Vwave:
    
    def __init__(self, pars, traj=Traj(), init_n=None, init_nh=None):
        self.p = pars
        self.n_steps = int((pars.tot_time + pars.t_burn)/pars.dt)
        self.traj = traj
        self.init_n, self.init_nh = init_n, init_nh
        self.is_stoch = False
        
        
    def run(self):
        
        par = self.p
        self.traj.start(self)
        self._start()
        
        if self.p.verbose:            
            print(' ', end='', flush=True)
            progress = tqdm(desc=f"Process {par.id}", total=self.n_steps, position = par.id)
            
        for t in range(self.n_steps):

            self._step(t)

            if not par.traj_after_burn or (par.traj_after_burn and t*par.dt > par.t_burn):
                if t % par.traj_step == 0:
                    self.traj.update(t)

            if par.check_step > 0 and t*par.dt > par.t_burn and t % par.check_step == 0:
                self._check_wave()

            if self.p.verbose:  
                progress.update(1) 

        return self
    
    
    def _start(self):
        
        par = self.p
        B = par.n_x_bins
        x0 = int(B/4)
        xvals = np.arange(0, B)
        
        self.cutoff = 0
        
        # Storing kernel fourier transform
        self.fft_kernel = np.fft.fft(1 - np.exp(-abs(xvals - int(B/2))*par.dx/par.r0))
        
        # Initializing the population of infcted hosts
        if type(self.init_n) != np.ndarray:
            self._init_n(B, x0, xvals)
        else:
            self.n = self._init_n_by_prior()
            x0 = np.argmax(self.n)
        
        # Initializing the immune system protection
        if type(self.init_nh) != np.ndarray:
            vtau = ut.vtau_lin(par)
            heaviside = np.ones(B)
            heaviside[x0:] = 0
            nh = np.exp(-np.abs((xvals - x0)*par.dx)/(vtau)) * heaviside / vtau
            nh = par.Nh * par.M * nh / (nh.sum()*par.dx)
            self.nh = nh.astype(np.float32)
        else:
            self.nh = self.init_nh.astype(np.float64)
    
        # Updating the wave positions
        self.back_i = nbf.find_last_zero(self.n, x0, 1e-10)
        self.front_i = nbf.find_first_zero(self.n, x0, 1e-10)
        self.max_i = nbf.find_max(self.n, x0)
        self.fronth_i = nbf.find_first_zero(self.nh, x0, 1e-10)
        self.kernel_range_i = int(par.r0*10/par.dx)
        
    
    # Function that will be overrided by the derived classes for evolution
    def _init_n(self, B, x0, xvals):
        n = np.zeros(B)
        x_range = ((xvals - x0)*self.p.dx - 0.2*self.p.r0) / (0.6*self.p.r0)
        n = ut.skewed_gaussian(x_range, 4)
        n = self.p.N0 * n / (n.sum()*self.p.dx)
        self.n = n.astype(np.float64)
        
        
    def _init_n_by_prior(self):
        self.n = self.init_n.astype(np.float64)        
        
        
    def _step(self, t):
        p = self.p
        
        if t*p.dt > p.t_burn_cutoff:
            self.cutoff = p.cutoff
        
        # Updating wave positions
        self.back_i = nbf.find_last_zero(self.n, self.back_i, 1e-10)
        self.front_i = nbf.find_first_zero(self.n, self.front_i, 1e-10)
        self.max_i = nbf.find_max(self.n, self.max_i)
        
        # Computing prob of infection through convolution
        fft_nh = np.fft.fft(self.nh/(self.nh.sum()*p.dx))
        self.P = np.fft.fftshift( np.real( np.fft.ifft( fft_nh*self.fft_kernel ) ) )*p.dx
        
        # Updating host population using numba function
        self._update_n()
        
        # Updating immune system
        self.nh += (np.float32(self.n) - np.float32(self.n.sum()*p.dx)*self.nh/(p.M*p.Nh))*p.dt
        
        # Killing possible trails of nh generated by the boundary conditions
        self.fronth_i = nbf.find_first_zero(self.nh, self.fronth_i, 1e-10)
        if self.fronth_i+self.kernel_range_i < p.n_x_bins:
            self.nh[self.fronth_i:self.fronth_i+self.kernel_range_i] = np.zeros(self.kernel_range_i)
        else:
            i1 = self.fronth_i+self.kernel_range_i-p.n_x_bins
            self.nh[:i1] = np.zeros(i1)
            self.nh[self.fronth_i:] = np.zeros(p.n_x_bins-self.fronth_i)
            
            
    # Function that will be overrided by the derived classes for evolution
    def _update_n(self):
        p = self.p
        self.n = nbf.compute_new_n(self.n, self.P, p.D_coef, p.dt, p.dx, p.alpha, p.beta, p.gamma, p.M,
                                  self.cutoff, p.back_width_fract, self.back_i, self.max_i, self.front_i)
        
        
    def _check_wave(self):
        
        N_coef = np.abs(self.traj.nvirus[-1] - self.traj.nvirus[-2])/self.p.N0
        if N_coef > 1e-2:
            print("proc.", self.p.id, " WARNING on stationaity of N,", N_coef)
            
        w_coef = np.abs(self.traj.width()[-1] - self.traj.width()[-2])
        if np.abs(self.traj.width()[-1] - self.traj.width()[-2]) > 2*self.p.dx:
            print("proc.", self.p.id, " WARNING on stationaity of width, ", w_coef)
        
        w_res_coef = self.traj.width()[-1]/self.p.dx
        if w_res_coef < 100:
            print("proc.", self.p.id, " WARNING on wave width resolution (increase dx), ", w_res_coef)
        
        nh_max_i = nbf.find_max(self.nh, self.fronth_i)
        nh_coef = self.nh[(self.fronth_i+self.kernel_range_i+1)%self.p.n_x_bins]/self.nh[nh_max_i]
        if nh_coef > 1e-3:
            print("proc.", self.p.id, " WARNING on nh approximation (increase x domain), ", nh_coef)
            
        
class Vwave_stoch_lang(Vwave):

    def __init__(self, p, traj=Traj()):
        Vwave.__init__(self, p, traj)
        self.is_stoch = True
        
    def _update_n(self):
        p = self.p
        self.n = nbf.compute_n_stochastic_lang(self.n, self.P, p.D_coef, p.dt, p.dx, p.alpha, p.beta, p.gamma, p.M,
                                  self.cutoff, p.back_width_fract, self.back_i, self.max_i, self.front_i)
        
class Vwave_stoch_poiss(Vwave):

    def __init__(self, p, traj=Traj()):
        Vwave.__init__(self, p, traj)
        self.is_stoch = True
        
    def _update_n(self):
        p = self.p
        self.n = nbf.compute_n_stochastic_poiss(self.n, self.P, p.D_coef, p.dt, p.dx, p.alpha, p.beta, p.gamma, p.M,
                                  self.cutoff, p.back_width_fract, self.back_i, self.max_i, self.front_i)
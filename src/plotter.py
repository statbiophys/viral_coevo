import matplotlib.pyplot as plt
import numpy as np
import numba_functions as nb
import utils as ut
import wave as w
import wave_virulence as wv
import wave_mut_rate as wmr


        
def plot_populations(ax1, wave):
    
    dx = wave.p.dx
    #ax1.set_xlim(nb.find_last_zero(wave.nh, 500, 1000)*dx, wave.front_i*dx)
    ax2 = ax1.twinx()
    ax1.set_xlabel('antigenic coord, x')
    ax1.set_ylabel('infected population')
    ax2.set_ylabel('immune system population')
    ax2.set_ylim(1, max(wave.nh))
    
    ax1.plot(np.arange(len(wave.n))*dx, wave.n, color = 'b')
    ax2.plot(np.arange(len(wave.n))*dx, wave.nh, color = 'r')
    
    return ax1, ax2


def plot_fitness(ax1, wave, space_outside_wave=30):
    
    p = wave.p
    left, right = wave.back_i-space_outside_wave, wave.front_i+space_outside_wave

    n, P = wave.n[left:right], wave.P[left:right]
    if left > right:
        n = np.append(wave.n[left:], wave.n[:right])
        P = np.append(wave.P[left:], wave.P[:right])
        right += len(wave.n)
    xvals = (np.arange(left, right) - wave.max_i)*p.dx
        
    if type(wave) == wv.Vwave_virulence:
        s = ut.sel_coef_lin(p, ut.vtau_lin_virulence(p))
        f = p.beta*np.sqrt(p.alpha)*P**p.M - (p.alpha + p.gamma)
    if type(wave) == wmr.Vwave_mut_rate:
        s = ut.sel_coef_lin(p, ut.vtau_lin_virulence(p))
        d = np.mean(wave.traj.mean_D[-50:])
        f = p.beta*P**p.M - (p.alpha + p.gamma + p.lambda_tilde*d)
    else:
        s = ut.sel_coef_lin(p, ut.vtau_lin(p))
        f = p.beta*P**p.M - (p.alpha + p.gamma)
    
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('antigenic coord, x')
    ax1.set_ylabel('infected population')
    ax2.set_ylabel('fitness')
    
    ax1.set_ylim(0.1, max(wave.n))
    ax2.set_ylim(min(f), max(f))
    ax1.set_xlim(xvals[0], xvals[-1])
    ax1.plot(xvals, n, '-o', c='b')
    ax2.plot(xvals, f, c='r')
    ax2.plot(xvals, s*xvals)
    ax2.axvline(x = 0, linestyle = '--', color = 'k')
    ax1.axhline(y = p.cutoff, linestyle = '--', color = 'b')
    ax1.set_yscale('log')
    
    return ax1, ax2


def plot_fitness_2waves(ax1, wave, space_outside_wave=30):
    ax11, ax12 = plot_fitness(ax1, wave, space_outside_wave)
    left, right = wave.back_i-space_outside_wave, wave.front_i+space_outside_wave
    n2 = wave.n2[1,left:right]
    if left > right:
        n2 = np.append(wave.n2[1,left:], wave.n2[1,:right])
        right += len(wave.n)
    xvals = (np.arange(left, right) - wave.max_i)*wave.p.dx   
    ax11.plot(xvals, n2, '-o', c='g')
    return ax11, ax12


def plot_trajectories(ax1, ax2, ax3, wave, after_burn=False):
           
    i0 = 0
    if after_burn:
        i0 = int(wave.p.t_burn/wave.p.dt/wave.p.traj_step)
        
    speed = wave.traj.speed()
    h_times = (wave.traj.times[1:] + wave.traj.times[:-1])/2
    ax1.set_xlabel('time')
    ax1.set_ylabel('wave speed')
    ax1.plot(h_times[i0:], speed[i0:])

    ax2.set_xlabel('time')
    ax2.set_ylabel('population size, N')
    ax2.plot(wave.traj.times[i0:], wave.traj.nvirus[i0:])

    ax3.set_xlabel('time')
    ax3.set_ylabel('wave width')
    ax3.plot(wave.traj.times[i0:], wave.traj.width()[i0:])
    
    if not after_burn:
        ax1.axvline(x = wave.p.t_burn, linestyle = '--', color = 'k')
        ax1.axvline(x = wave.p.t_burn_cutoff, linestyle = '--', color = 'k')
        ax2.axvline(x = wave.p.t_burn, linestyle = '--', color = 'k')
        ax2.axvline(x = wave.p.t_burn_cutoff, linestyle = '--', color = 'k')
        ax3.axvline(x = wave.p.t_burn, linestyle = '--', color = 'k')
        ax3.axvline(x = wave.p.t_burn_cutoff, linestyle = '--', color = 'k')
    
    return ax1, ax2, ax3


def plot_virulence_trajectory(ax, wave, after_burn=False):
    
    i0 = 0
    if after_burn:
        i0 = int(wave.p.t_burn/wave.p.dt/wave.p.traj_step)
        
    mean_a = wave.traj.mean_alpha[i0:]
    std_a = wave.traj.std_alpha[i0:]
    
    ax.set_xlabel('time')
    ax.set_ylabel('alpha')
    ax.fill_between(wave.traj.times[i0:], mean_a-std_a, mean_a+std_a, alpha=0.5)
    ax.plot(wave.traj.times[i0:], mean_a)
    
    if not after_burn:
        ax.axvline(x = wave.p.t_burn, linestyle = '--', color = 'k')
        ax.axvline(x = wave.p.t_burn_cutoff, linestyle = '--', color = 'k')
    
    return ax


def av_smoother(traj, n_smooth_bins):
    return np.array([np.mean(np.array(traj)[i-n_smooth_bins+1:i+1]) for i in range(n_smooth_bins, len(traj))])
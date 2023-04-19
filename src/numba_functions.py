import numba as nb
import numpy as np


@nb.jit(nopython=True, fastmath=True)
def find_last_zero(v, start_index, effective_zero=0):
    """
    Find the index in vector who is zero and its right element is non-zero
    starting from start_index
    """
    l = len(v)
    i = start_index%l
    no_zero_found, zero_found = False, False
    for _ in range(l):
        if v[i] > effective_zero:
            if zero_found:
                return i-1
            i = (i-1)%l
            no_zero_found = True
        else:
            if no_zero_found:
                return i
            else:
                i = (i+1)%l
                zero_found = True
    raise Exception('Zero not found in the vector')

    
@nb.jit(nopython=True, fastmath=True)
def find_first_zero(v, start_index, effective_zero=0):
    """
    Find the index in vector who is zero and its left element is non-zero
    starting from start_index
    """
    l = len(v)
    i = start_index%l
    no_zero_found, zero_found = False, False
    for _ in range(l):
        if v[i] > effective_zero:
            if zero_found:
                return i+1
            i = (i+1)%l
            no_zero_found = True
        else:
            if no_zero_found:
                return i
            else:
                i = (i-1)%l
                zero_found = True
    raise Exception('Zero not found in the vector')
    
    
@nb.jit(nopython=True, fastmath=True)
def find_max(v, start_index):
    """
    Find the closest local maximum around start_index
    """
    l = len(v)    
    old_i = start_index%l
    i = (old_i+1)%l
    go_sign = int(2*(int(v[i] > v[old_i])-0.5)) 
    for _ in range(l):
        old_i = i
        i = (old_i+go_sign)%l
        if v[i] < v[old_i]:
            return old_i
        
    raise Exception('Zero not found in the vector')
    
    
@nb.jit(nopython=True, fastmath=True)
def is_in_periodic_range(i, left, right):
    """
    Check whether i is in between left( and right] in an interval
    with boundary conditions
    """
    if (right > left):
        if ((i > left) & (i <= right)):
            return True
        else:
            return False
    else:
        if ((i > left) | (i <= right)):
            return True
        else:
            return False
        

@nb.jit(nopython=True, fastmath=True)
def compute_new_n(n, P, D, dt, dx, alpha, beta, gamma, M, cutoff, back_width_fr, back_i, max_i, front_i):
    """ Update one temporal step of the wave equation """
    
    B = len(n)        
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B

    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))
        
    uyy = np.zeros(B, dtype = np.float64)
    urate = np.zeros(B, dtype = np.float64)
    d_n = np.zeros(B)

    for jx in wave_indexes:
            
        uyy[jx] = n[(jx-1)%B] + n[(jx+1)%B] - 2*n[jx] 
        urate[jx] = beta*P[jx]**M - alpha - gamma

        is_front = is_in_periodic_range(jx, max_i, front_i)
        if ((n[jx] < cutoff) & is_front):
            urate[jx] = 0

        is_back = is_in_periodic_range(jx, back_i_update, max_i)
        if ((~is_back) & (~is_front)):
            urate[jx] = - alpha - gamma

        d_n[jx] = d_n[jx] + dt*(D*uyy[jx]/dx**2 + urate[jx]*n[jx])

    return n + d_n


@nb.jit(nopython=True, fastmath=True)
def compute_n_stochastic_lang(n, P, D, dt, dx, alpha, beta, gamma, M, cutoff, back_width_fr, back_i, max_i, front_i):   
    # cutoff is now the stochastic cutoff below which stocasticity is applied
    """ Update one temporal step of the wave equation with Langevin stochastic rule at the tip """
    
    B = len(n)    
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B
        
    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))
        
    uyy = np.zeros(B, dtype = np.float64)    
    for jx in wave_indexes:
        uyy[jx] = n[(jx-1)%B] + n[(jx+1)%B] - 2*n[jx] 

    for jx in wave_indexes:
            
        is_front = is_in_periodic_range(jx, max_i, front_i)
        is_back = is_in_periodic_range(jx, back_i_update, max_i)
        
        if ((~is_back) & (~is_front)):
             urate = - alpha - gamma
        else:
             urate = beta*P[jx]**M - alpha - gamma
        
        dn = dt*(D*uyy[jx]/dx**2 + urate*n[jx])
        
        if ((n[jx]*dx < cutoff) & is_front):
            dn += np.sqrt(n[jx]*dt)*np.random.rand()
            n[jx] = max(0, int((n[jx] + dn)*dx)/dx)
        else:
            n[jx] = n[jx] + dn
        
        
    return n


@nb.jit(nopython=True, fastmath=True)
def compute_n_stochastic_poiss(n, P, D, dt, dx, alpha, beta, gamma, M, cutoff, back_width_fr, back_i, max_i, front_i):   
    # cutoff is now the stochastic cutoff below which stocasticity is applied
    """ Update one temporal step of the wave equation with a poisson process at the tip """
    
    B = len(n)    
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B
        
    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))
        
    uyy = np.zeros(B, dtype = np.float64)
    mut_factor = 1 - np.exp(-2*D*dt/dx**2)
    
    for jx in wave_indexes:
            
            is_front = is_in_periodic_range(jx, max_i, front_i)
            if ((n[jx]*dx < cutoff) & is_front):
                n_mut = np.random.binomial(int(n[jx]*dx), mut_factor)
                n_neighs = np.random.multinomial(n_mut, np.ones(2)/2).astype(np.float64)
            else:
                n_mut = 2*D*n[jx]*dt/dx
                n_neighs = np.ones(2)*n_mut/2

            uyy[jx] += - n_mut/dx
            uyy[(jx-1)%B] += n_neighs[0]/dx
            uyy[(jx+1)%B] += n_neighs[1]/dx

    for jx in wave_indexes:
            
        is_front = is_in_periodic_range(jx, max_i, front_i)
        is_back = is_in_periodic_range(jx, back_i_update, max_i)
        
        if ((~is_back) & (~is_front)):
             urate = - alpha - gamma
        else:
             urate = beta*P[jx]**M - alpha - gamma
        
        n[jx] += uyy[jx]
        
        n_av = (1 + np.maximum(-1, urate*dt))*n[jx]*dx
        if ((n_av < cutoff) & is_front):
            n[jx] = np.random.poisson(n_av)/dx
        else:
            n[jx] = n_av/dx
        
    return n


@nb.jit(nopython=True, fastmath=True)
def compute_n_alpha(n_alpha, n, P, alpha_neighs, D, eps, dt, dx, beta0, gamma, alphas, dalpha, M, cutoff, back_width_fr, 
                    back_i, max_i, front_i, global_cutoff):
   
    A = n_alpha.shape[0]
    B = n_alpha.shape[1]
    
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B
        
    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))
    
    for imut in range(A): 
        
        alpha = alphas[imut]
        beta = beta0*np.sqrt(alpha)
        
        _alpha_neighs = alpha_neighs[imut,:]
        _alpha_neighs = np.delete(_alpha_neighs, np.argwhere(_alpha_neighs==-1).flatten())
        
        uxx = np.zeros(B, dtype = np.float64)
        uyy = np.zeros(B, dtype = np.float64)
        urate = np.zeros(B, dtype = np.float64)
        d_n_alpha = np.zeros(B)

        if(A > 1):
            for jx in wave_indexes:
        
                zmu = len(_alpha_neighs)
                for neigh in _alpha_neighs:
                    uxx[jx] = uxx[jx] + n_alpha[neigh,jx]

                uxx[jx] = uxx[jx] - zmu*n_alpha[imut,jx]
                d_n_alpha[jx] += dt*eps*uxx[jx]/dalpha**2
            
        for jx in wave_indexes:

            uyy[jx] = n_alpha[imut, (jx-1)%B] + n_alpha[imut, (jx+1)%B] - 2*n_alpha[imut, jx] 
            urate[jx] = beta*P[jx]**M - (alpha + gamma)

            is_front = is_in_periodic_range(jx, max_i, front_i)
            if (((global_cutoff & (n[jx] < cutoff)) | ((~global_cutoff) & (n_alpha[imut,jx] < cutoff))) & is_front):
                    urate[jx] = 0

            is_back = is_in_periodic_range(jx, back_i_update, max_i)
            if ((~is_back) & (~is_front)):
                urate[jx] = - alpha - gamma

            d_n_alpha[jx] += dt*(D*uyy[jx]/dx**2 + urate[jx]*n_alpha[imut, jx])
        
        n_alpha[imut,:] = n_alpha[imut,:] + d_n_alpha
        
    return n_alpha


@nb.jit(nopython=True, fastmath=True)
def compute_n_alpha_stoch(n_alpha, n, P, alpha_neighs, D, eps, dt, dx, beta0, gamma, alphas, dalpha, M, cutoff, back_width_fr, 
                          back_i, max_i, front_i, global_cutoff):
   
    A = n_alpha.shape[0]
    B = n_alpha.shape[1]
    
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B
        
    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))
    
    mut_factor = 1 - np.exp(-2*D*dt/dx**2)
    
    for imut in range(A): 
        
        alpha = alphas[imut]
        beta = beta0*np.sqrt(alpha)
        
        _alpha_neighs = alpha_neighs[imut,:]
        _alpha_neighs = np.delete(_alpha_neighs, np.argwhere(_alpha_neighs==-1).flatten())
        
        uyy = np.zeros(B, dtype = np.float64)
        urate = np.zeros(B, dtype = np.float64)

        if(A > 1):
            for jx in wave_indexes:
        
                zmu = len(_alpha_neighs)
                uxx = 0
                for neigh in _alpha_neighs:
                    uxx += n_alpha[neigh,jx]

                uxx = uxx - zmu*n_alpha[imut,jx]
                n_alpha[imut, jx] += dt*eps*uxx/dalpha**2
                
        for jx in wave_indexes:
            is_front = is_in_periodic_range(jx, max_i, front_i)
            
            if (((global_cutoff & (n[jx]*dx < cutoff)) | (~global_cutoff & (n_alpha[imut,jx]*dx < cutoff))) & is_front):
                n_mut = np.random.binomial(int(n_alpha[imut,jx]*dx), mut_factor)
                n_neighs = np.random.multinomial(n_mut, np.ones(2)/2).astype(np.float64)
            else:
                n_mut = 2*D*n_alpha[imut,jx]*dt/dx
                n_neighs = np.ones(2)*n_mut/2

            uyy[jx] += - n_mut/dx
            uyy[(jx-1)%B] += n_neighs[0]/dx
            uyy[(jx+1)%B] += n_neighs[1]/dx
            

        for jx in wave_indexes:

            is_front = is_in_periodic_range(jx, max_i, front_i)
            is_back = is_in_periodic_range(jx, back_i_update, max_i)

            if ((~is_back) & (~is_front)):
                 urate = - alpha - gamma
            else:
                 urate = beta*P[jx]**M - alpha - gamma

            n_alpha[imut,jx] += uyy[jx]

            n_av = (1 + np.maximum(-1, urate*dt))*n_alpha[imut,jx]*dx
            
            if (((global_cutoff & (n[jx]*dx < cutoff)) | (~global_cutoff & (n_alpha[imut,jx]*dx < cutoff))) & is_front):
                n_alpha[imut,jx] = np.random.poisson(n_av)/dx
            else:
                n_alpha[imut,jx] = n_av/dx
        
    return n_alpha



@nb.jit(nopython=True, fastmath=True)
def compute_n_D(n_D, n, P, D_neighs, eps, dt, dx, alpha, beta, gamma, lambd, Ds, dD, M, cutoff, back_width_fr, 
                    back_i, max_i, front_i, global_cutoff):
   
    A = n_D.shape[0]
    B = n_D.shape[1]
    
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B
        
    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))
    
    for imut in range(A): 
        
        D = Ds[imut]
        #b_eff = beta * np.exp(-lambd * D)
        
        _D_neighs = D_neighs[imut,:]
        _D_neighs = np.delete(_D_neighs, np.argwhere(_D_neighs==-1).flatten())
        
        uxx = np.zeros(B, dtype = np.float64)
        uyy = np.zeros(B, dtype = np.float64)
        urate = np.zeros(B, dtype = np.float64)
        d_n_D = np.zeros(B)

        if(A > 1):
            for jx in wave_indexes:
        
                zmu = len(_D_neighs)
                for neigh in _D_neighs:
                    uxx[jx] = uxx[jx] + n_D[neigh,jx]

                uxx[jx] = uxx[jx] - zmu*n_D[imut,jx]
                d_n_D[jx] += dt*eps*uxx[jx]/dD**2
            
        for jx in wave_indexes:

            uyy[jx] = n_D[imut, (jx-1)%B] + n_D[imut, (jx+1)%B] - 2*n_D[imut, jx] 
            urate[jx] = beta*P[jx]**M * (1 - lambd * D) - (alpha + gamma)

            is_front = is_in_periodic_range(jx, max_i, front_i)
            if (is_front):
                if ((global_cutoff & (n[jx] < cutoff)) | ((~global_cutoff) & (n_D[imut,jx] < cutoff))):
                    urate[jx] = 0

            is_back = is_in_periodic_range(jx, back_i_update, max_i)
            if ((~is_back) & (~is_front)):
                urate[jx] = - alpha - gamma

            d_n_D[jx] += dt*(D*uyy[jx]/dx**2 + urate[jx]*n_D[imut, jx])
        
        n_D[imut,:] = n_D[imut,:] + d_n_D
        
    return n_D


@nb.jit(nopython=True, fastmath=True)
def compute_n_D_stoch(n_D, n, P, D_neighs, eps, dt, dx, alpha, beta, gamma, lambd, Ds, dD, M, cutoff, back_width_fr, 
                    back_i, max_i, front_i, global_cutoff):
   
    A = n_D.shape[0]
    B = n_D.shape[1]
    
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B
        
    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))
    
    for imut in range(A): 
        
        D = Ds[imut]
        mut_factor = 1 - np.exp(-2*D*dt/dx**2)
        #b_eff = beta * np.exp(-lambd * D)
        
        _D_neighs = D_neighs[imut,:]
        _D_neighs = np.delete(_D_neighs, np.argwhere(_D_neighs==-1).flatten())
        
        uyy = np.zeros(B, dtype = np.float64)
        urate = np.zeros(B, dtype = np.float64)

        if(A > 1):
            for jx in wave_indexes:
        
                zmu = len(_D_neighs)
                uxx = 0
                for neigh in _D_neighs:
                    uxx += n_D[neigh,jx]

                uxx = uxx - zmu*n_D[imut,jx]
                n_D[imut, jx] += dt*eps*uxx/dD**2
                
        for jx in wave_indexes:
            is_front = is_in_periodic_range(jx, max_i, front_i)
            
            if (((global_cutoff & (n[jx]*dx < cutoff)) | (~global_cutoff & (n_D[imut,jx]*dx < cutoff))) & is_front):
                n_mut = np.random.binomial(int(n_D[imut,jx]*dx), mut_factor)
                n_neighs = np.random.multinomial(n_mut, np.ones(2)/2).astype(np.float64)
            else:
                n_mut = 2*D*n_D[imut,jx]*dt/dx
                n_neighs = np.ones(2)*n_mut/2

            uyy[jx] += - n_mut/dx
            uyy[(jx-1)%B] += n_neighs[0]/dx
            uyy[(jx+1)%B] += n_neighs[1]/dx
            

        for jx in wave_indexes:

            is_front = is_in_periodic_range(jx, max_i, front_i)
            is_back = is_in_periodic_range(jx, back_i_update, max_i)

            if ((~is_back) & (~is_front)):
                 urate = - alpha - gamma
            else:
                 urate = beta*P[jx]**M * (1 - lambd * D) - (alpha + gamma)

            n_D[imut,jx] += uyy[jx]

            n_av = (1 + np.maximum(-1, urate*dt))*n_D[imut,jx]*dx
            
            if (((global_cutoff & (n[jx]*dx < cutoff)) | (~global_cutoff & (n_D[imut,jx]*dx < cutoff))) & is_front):
                n_D[imut,jx] = np.random.poisson(n_av)/dx
            else:
                n_D[imut,jx] = n_av/dx
        
    return n_D



@nb.jit(nopython=True, fastmath=True)
def compute_n2(n2, P, is_mut_flux, is_init, mut_flux, D, D_m, dt, dx, beta, beta_m, gamma, gamma_m, alpha, alpha_m, M, cutoff, back_width_fr, back_i, max_i, front_i):
   
    B = n2.shape[1]
    
    back_i, front_i = (back_i-1)%B, (front_i+1)%B
    back_i_update = int(max_i - (front_i-max_i)*back_width_fr)%B
    if front_i < max_i:
        back_i_update = int(max_i - (front_i-max_i+B)*back_width_fr)%B
        
    wave_indexes = np.arange(back_i, front_i)
    if front_i < back_i:
        wave_indexes = np.append(np.arange(back_i, B), np.arange(0, front_i))

    uyy = np.zeros(B, dtype = np.float64)
    urate = np.zeros(B, dtype = np.float64)
    d_n = np.zeros(B)

    for jx in wave_indexes:
            
        uyy[jx] = n2[0,(jx-1)%B] + n2[0,(jx+1)%B] - 2*n2[0,jx] 
        urate[jx] = beta*P[jx]**M - alpha - gamma
        #if (is_init & is_mut_flux):
        #    urate[jx] -= mut_flux

        is_front = is_in_periodic_range(jx, max_i, front_i)
        if ((n2[0,jx]+n2[1,jx] < cutoff) & is_front):
            urate[jx] = 0

        is_back = is_in_periodic_range(jx, back_i_update, max_i)
        if ((~is_back) & (~is_front)):
            urate[jx] = - alpha - gamma

        d_n[jx] = d_n[jx] + dt*(D*uyy[jx]/dx**2 + urate[jx]*n2[0,jx])
    
    if (is_init):
        d_n2 = np.zeros(B)
        
        for jx in wave_indexes:

            uyy[jx] = n2[1,(jx-1)%B] + n2[1,(jx+1)%B] - 2*n2[1,jx] 
            urate[jx] = (beta_m*P[jx]**M - alpha_m - gamma_m)*n2[1,jx]
            if (is_mut_flux):
                urate[jx] += mut_flux*n2[0,jx]
            
            is_front = is_in_periodic_range(jx, max_i, front_i)
            if ((n2[0,jx]+n2[1,jx] < cutoff) & is_front):
                urate[jx] = 0

            is_back = is_in_periodic_range(jx, back_i_update, max_i)
            if ((~is_back) & (~is_front)):
                urate[jx] = (- alpha - gamma)*n2[1,jx]

            d_n2[jx] = d_n2[jx] + dt*(D_m*uyy[jx]/dx**2 + urate[jx])

        n2[1,:] = n2[1,:] + d_n2
    
    n2[0,:] = n2[0,:] + d_n
    return n2
"""
Shared functions for "Bridge change of measure for propagator estimation".

Contains the exact propagators (OU, CIR, Wiener), the bridge samplers
(Wiener and Ornstein-Uhlenbeck bridges), the BCMMC estimators with Wiener and
OU bridges, the best-in-class OU-bridge parameters, the crude Monte Carlo
reference estimator and the plotting style used in the paper figures.
"""
import matplotlib
import numpy as np
from numpy import exp as exp
from numpy import log as log
from numpy import sqrt as sqrt
from numpy import sinh as sinh
from scipy.special import ive
from scipy.special import gammaln
import matplotlib.pyplot as plt
from numba import njit
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams['axes.linewidth']=1 #Mattia had 2 -> Grosor del marco (doble del standard)

def Plot_bonito(xlabel=r" $ x$",ylabel=r"$ y$",label_font_size=15,ticks_size=12,y_size=2.4,x_size=3.2):
    plt.figure(figsize=(x_size,y_size))
    plt.tick_params(labelsize=24)
    plt.xlabel(xlabel,fontsize=label_font_size)
    plt.ylabel(ylabel,fontsize=label_font_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.locator_params(axis="both", nbins=5,tight=True)

def axis_bonito(ax,xlabel=r" $ x$",ylabel=r"$ y$",label_font_size=15,ticks_size=12):
    ax.set_xlabel(xlabel,fontsize=label_font_size)
    ax.set_ylabel(ylabel,fontsize=label_font_size)
    ax.tick_params(axis="x", labelsize=ticks_size)
    ax.tick_params(axis="y", labelsize=ticks_size)
    # For two column figure that fits in one column of a two-columns paper:
    # fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(4.2,1.8))
    # For three rows figure without space between plots:
    # fig, (ax1,ax2,ax3) = plt.subplots(3, 1,sharex=True,figsize=(2,3.6))
    # fig.tight_layout()
    # -----plots--------
    # ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.subplots_adjust(hspace=0)

@njit
def Fill_gaps_with_Wiener_bridges(
    tf,
    t0=0,
    x0=0,
    xf=0,
    dt=-1,
    npoints=-1,
):
    """
    Simulate a Wiener (Brownian) bridge on [t0, tf] from x0 to xf.

    You must provide exactly one of:
        - dt      : time step
        - npoints : number of time points

    The other quantity is computed automatically.

    Parameters
    ----------
    tf : float
        Final time.
    t0 : float, optional
        Initial time.
    x0 : float, optional
        Initial value.
    xf : float, optional
        Final value.
    dt : float, optional
        Time step.
    npoints : int, optional
        Number of grid points (including endpoints).

    Returns
    -------
    ts : np.ndarray
        Time grid from t0 to tf.
    xs : np.ndarray
        Bridge values X_t at those times.
    dt : float
        Actual time step used.
    npoints : int
        Number of points used.
    """

    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    # Require exactly one of dt or npoints
    if (dt == -1 and npoints == -1) or (dt != -1 and npoints != -1):
        raise ValueError("Provide exactly one of dt or npoints")

    # Compute missing quantity
    if dt != -1:
        if dt <= 0:
            raise ValueError("dt must be positive")

        npoints = int(np.round((tf - t0) / dt)) + 1

        if npoints < 2:
            raise ValueError("dt is too large for the interval")

        # Recompute dt so endpoint is exact
        dt = (tf - t0) / (npoints - 1)

    else:
        if npoints < 2:
            raise ValueError("npoints must be at least 2")

        dt = (tf - t0) / (npoints - 1)

    ts = np.linspace(t0, tf, npoints)
    xs = np.zeros(npoints)
    xs[0] = x0
    xs[-1] = xf
    x = x0

    for i in range(1, npoints - 1):
        s = ts[i - 1]
        t = ts[i]
        h = t - s
        denom = tf - s
        mean = x + h * (xf - x) / denom
        var = h * (tf - t) / denom
        std = np.sqrt(var)
        x = mean + std * np.random.normal()
        xs[i] = x

    return ts, xs


@njit
def Setup_bridge_grid(tf, t0=0, dt=-1, npoints=-1):
    """
    Check parameters and build the time grid for a bridge on [t0, tf].

    You must provide exactly one of:
        - dt      : time step
        - npoints : number of time points

    The other quantity is computed automatically.

    Returns
    -------
    xs : np.ndarray
        Zero-initialized array to be filled in by Simulate_Wiener_bridge_path.
    ts : np.ndarray
        Time grid from t0 to tf.
    """

    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    # Require exactly one of dt or npoints
    if (dt == -1 and npoints == -1) or (dt != -1 and npoints != -1):
        raise ValueError("Provide exactly one of dt or npoints")

    # Compute missing quantity
    if dt != -1:
        if dt <= 0:
            raise ValueError("dt must be positive")

        npoints = int(np.round((tf - t0) / dt)) + 1

        if npoints < 2:
            raise ValueError("dt is too large for the interval")

        # Recompute dt so endpoint is exact
        dt = (tf - t0) / (npoints - 1)

    else:
        if npoints < 2:
            raise ValueError("npoints must be at least 2")

        dt = (tf - t0) / (npoints - 1)

    ts = np.linspace(t0, tf, npoints)
    xs = np.zeros(npoints)

    return xs, ts

@njit
def Simulate_OU_bridge_path(ts, xs, tf, x0, xf,mu,k):
    """Fill in a Ornstein-Uhlenbeck bridge path on the grid (ts, xs) from x0 to xf.
    Bridge of the process dX_t = k(mu-X_t) dt + dW_t (Eqs. G4-G5)."""
    npoints = len(ts)
    xs[0] = x0
    xs[-1] = xf
    x = x0

    for i in range(1, npoints - 1):
        s = ts[i - 1]
        t = ts[i]
        h = t - s
        denom = sinh(k * (tf - s))
        mean = mu + ((x - mu) * sinh(k * (tf - t)) + (xf - mu) * sinh(k * h)) / denom
        var = sinh(k * (tf - t)) * sinh(k * h) / (k * denom)
        std = np.sqrt(var)
        x = mean + std * np.random.normal()
        xs[i] = x

    return ts, xs

@njit
def Simulate_Wiener_bridge_path(ts, xs, tf, x0, xf):
    """Fill in a Wiener bridge path on the grid (ts, xs) from x0 to xf."""
    npoints = len(ts)
    xs[0] = x0
    xs[-1] = xf
    x = x0

    for i in range(1, npoints - 1):
        s = ts[i - 1]
        t = ts[i]
        h = t - s
        denom = tf - s
        mean = x + h * (xf - x) / denom
        var = h * (tf - t) / denom
        std = np.sqrt(var)
        x = mean + std * np.random.normal()
        xs[i] = x

    return ts, xs

@njit
def gaussian(x, mean, variance):
    """Evaluate the Gaussian distribution at x with given mean and variance."""
    coeff = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coeff * np.exp(exponent)

@njit
def propagator_OU_process(x, t, x0, mu, K, D):
    """
    Compute the propagator (transition probability density) of the Ornstein-Uhlenbeck process, defined by the SDE:
        dX_t = K (mu - X_t) dt + D dW_t

    Parameters:
    - x: scalar or array of positions where to evaluate the propagator
    - t: time
    - x0: initial position
    - mu: mean reversion level
    - K: strength of the restoring force
    - D: diffusion coefficient

    Returns:
    - Propagator evaluated at x and time t.
    """
    if K == 0:
        return propagator_WI_process(x, t, x0, D)
    else:
        mean = mu + (x0 - mu) * np.exp(-K * t)
        variance = (D**2 / (2 * K)) * (1 - np.exp(-2 * K * t))
    return gaussian(x, mean, variance)


def propagator_BDMM(x, t, x0, mu, k, D):
    """Propagator for process dX_t=k (μ - X_t) dt + D sqrt{X_t
    }dWt, compute the transition probability density p(x, t | x0, 0) using the exact solution."""
    if np.any(x < 0):
        raise ValueError("All values in x must be >= 0.")
    if np.any(t <= 0):
        raise ValueError("t must be > 0.")
    if np.any(x0 < 0):
        raise ValueError("x0 must be >= 0.")

    x = np.asarray(x)

    c = 2 * k / (D**2 * (1 - exp(-k * t)))
    u = c * x0 * exp(-k * t)
    v = c * x
    q = 2 * k * mu / D**2 - 1

    z = 2 * np.sqrt(u * v)

    if not np.all(np.isfinite(z)):
        raise ValueError("z is not finite.")

    # scaled Bessel: ive(q, z) = exp(-|z|) * iv(q, z)
    bessel_scaled = ive(q, z)

    # Mirrors propagator_DE_process_njit: where the scaled Bessel underflows to exactly 0
    # (small z with large q), fall back to the zeroth-order small-z series for log(I_q(z))
    # instead of taking log(0). np.where evaluates both branches over the whole array, so the
    # divide-by-zero / log(0) warning at the masked-out entries is expected and suppressed.
    small_z = bessel_scaled <= 0
    safe_bessel = np.where(small_z, 1.0, bessel_scaled)  # avoids log(0) warning; unused where small_z
    log_Iq = np.where(
        small_z,
        q * log(0.5 * z) - gammaln(q + 1.0),
        log(safe_bessel) + np.abs(z),   # compensates exponential in ive
    )

    # if not np.all(np.isfinite(log_Iq)):
    #     raise ValueError("log_Iq is not finite.")

    # log formulation
    log_pdf = (
        log(c)
        - (u + v)
        + (q / 2.0) * (log(v) - log(u))
        + log_Iq
    )

    pdf = exp(log_pdf)

    return pdf

@njit
def propagator_WI_process(x, t, x0, D):
    """
    Compute the propagator (transition probability density) of the Wiener (Brownian motion) process.

    Parameters:
    - x: scalar or array of positions where to evaluate the propagator
    - t: time
    - x0: initial position
    - D: diffusion coefficient

    Returns:
    - Propagator evaluated at x and time t.
    """
    mean = x0
    variance = D**2 * t
    return gaussian(x, mean, variance)

@njit
def prop_OU_W_bridge(x, t, y, s, mu, k, D, NB, dt_B):
    """Compute the OU propagator using Wiener bridges and Girsanov's theorem."""
    NB = int(NB)
    Ls = np.zeros(NB)

    for ii in range(NB):
        ts_B, xs_B = Fill_gaps_with_Wiener_bridges(tf = t-s,dt = dt_B, x0 = y, xf = x)
        # OU drift
        b = -k * (xs_B - mu)

        # Girsanov exponent components (left-point rule)
        dx = np.diff(xs_B)
        dt = np.diff(ts_B)

        state_integral = (1.0 / D**2) * np.sum(b[:-1] * dx) #This could be computed exactly
        time_integral  = (1.0 / D**2) * np.sum(b[:-1]**2 * dt)

        log_weight = state_integral - 0.5 * time_integral
        weight = np.exp(log_weight)

        prop_WI = propagator_WI_process(x, t-s, y, D)  # Wiener propagator
        Ls[ii] = weight * prop_WI
    empirical_prop = np.mean(Ls)
    return empirical_prop, Ls


def Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D):
    """Update system from x0 to x=x(tf) using Milstein method or exact DEM when theta=0"""
    if tf <= t0:
        raise ValueError("tf must be > t0.")
    if x0 < 0:
        raise ValueError("x0 must be >= 0.")

    x = x0
    t = t0
    D2 = D*D

    if theta == 0.0:
        dt = tf - t0
        beta = -k
        alpha = mu*k
        eb = np.exp(beta*dt)

        if eb == 1.0:
            raise FloatingPointError("exp(beta*dt) too close to 1, unstable denominator.")

        lam = 2.0*beta/D2/(eb-1.0)
        nu = 2.0*alpha/D2 - 1.0

        if lam <= 0:
            raise FloatingPointError("lambda must be > 0.")

        n = np.random.poisson(lam*x*eb)

        shape = n + nu + 1.0
        if shape <= 0:
            raise FloatingPointError("Gamma shape parameter <= 0.")

        x = np.random.gamma(shape, 1.0/lam)
        t = tf

        if x < 0:
            raise ValueError("x became negative, which is not allowed in this model.")

        return x

    steps = int((tf-t0)/dt_int)
    sqdt = sqrt(dt_int)
    dtheta = (theta+1.0)/2.0

    for ii in range(steps):
        t = t + dt_int
        u = np.random.normal()
        xth = x**(theta)
        milstein = D2*dtheta*xth*dt_int/2.0
        drift = k*(mu-x)*xth*dt_int 
        noise = D*(x**(dtheta))*sqdt
        xnew = x + drift + noise*u + milstein*(u*u-1.0)
        x = xnew
        if x < 0:
            print("x=",x,"t=",t,"ii=",ii)
            raise ValueError("x became negative, which is not allowed in this model.")

    return x


def Estimate_propagator_crude_MC_on_point(x0,t0,tf,dt_int,realiz,x_target,dx,theta,mu,k,D):
    th = dx/2.0
    count = 0
    for ii in range(realiz):
        x =  Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D)
        if (x>(x_target-th))and(x<=(x_target+th)):
            count+=1
    prop = count/realiz/dx
    return prop

def Estimate_propagator_crude_MC(x0,t0,tf,dt_int,realiz,npoints,params,model,logscale = False):
    """Estimate the propagator on a grid of points using crude Monte Carlo."""
    """Models: 
                0 --> OU
                1 --> CIR // BDMM // DE
                2 --> SLM // EN
                3 --> General gamma model"""
    xs = np.zeros(realiz)
    if model == 1:
        mu, k, D = params
        theta = 0.0
        for ii in range(realiz):
            x =  Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D)
            xs[ii] = x
    elif model == 2:
        mu, k, D = params
        theta = 1.0
        for ii in range(realiz):
            x =  Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D)
            xs[ii] = x
    elif model == 3:
        mu, k, D, theta = params
        for ii in range(realiz):
            x =  Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D)
            xs[ii] = x
    else:
        print("Model not recognized in Estimate_propagator_crude_MC (Flag1)")
        return -1.0
    
    if logscale:
        bins = np.logspace(np.log10(min(xs)), np.log10(max(xs)), npoints+1)
    else:
        bins = np.linspace(min(xs), max(xs), npoints+1)
    hist, edges = np.histogram(xs, bins=bins, density=True)
    xs_mid = (edges[:-1] + edges[1:]) / 2
    return xs_mid, hist

@njit
def prop_BCM_OU_bridges(x, t, y, s, params, model, NB, dt_B, params_B, errors = False):
    """
    Compute the propagator of model using Ornstein-Uhlenbeck (OU) bridges and Bridge change of measure technique.
    The (unconditioned) OU process is given by the SDE:
        dX_t = k_OU(mu_OU - X_t) dt + dW_t
    Available models:
    0 --> Ornstein Uhlenbeck                          dX_t= k(mu-X_t) dt + D dW_t
    1 --> Cox-Ingersoll-Ross // Birth death migration dX_t= k(mu-X_t) dt + D sqrt(X_t) dW_t
    2 --> Stochastic logistic                         dX_t= k(mu-X_t)X_t dt + D X_t dW_t
    """
    NB = int(NB)
    Ls = np.zeros(NB)
    mu_OU, k_OU = params_B
    mu, k, D = params
    #1) Transform data. Same transformation as in prop_BCM_Wiener_bridges.
    if model == 0: #OU
        hx = x/D
        hy = y/D
        Jacobian = 1.0/D
    elif model == 1: #CIR // BDMM // DE
        hx = 2.0*sqrt(x)/D
        hy = 2.0*sqrt(y)/D
        Jacobian = 1.0/(D*sqrt(x))
    elif model == 2: #SLM
        hx = log(x)/D
        hy = log(y)/D
        Jacobian = 1.0/(D*x)
    else:
        print("ERROR: Model not recognized in prop_BCM_OU_bridges (Flag1)")
        return -1.0, np.zeros(NB), 0.0
    
    prop_OU = propagator_OU_process(hx, t-s, hy, mu_OU, k_OU, 1.0)  # OU propagator
    
    #2) Build the time grid for the Wiener bridges (only once)
    xs_B, ts_B = Setup_bridge_grid(tf = t-s, dt = dt_B)
    dt = np.diff(ts_B) 

    for ii in range(NB):
        #3) Simulate Wiener bridges
        ts_B, xs_B = Simulate_OU_bridge_path(ts_B, xs_B, tf = t-s, x0 = hy, xf = hx, mu = mu_OU, k = k_OU)
        #4) Compute drift of the transformed process
        if model == 0: #OU
            h_invs = D*xs_B
            der_h = np.full_like(h_invs, 1.0/D) #Due to njit, this cannot be a constant if in a different branch it is an array
            der_h2 = np.zeros_like(xs_B)
            drift = k * (mu-h_invs)
            noise2 = np.full_like(h_invs, D**2.0)
        elif model == 1: #CIR // BDMM // DE
            h_invs = (D*xs_B)**2.0/4.0
            der_h = 1.0/(D*sqrt(h_invs))
            der_h2 = -h_invs**(-1.5)/(2.0*D)
            drift = k * (mu-h_invs)
            noise2 = D**2.0 * h_invs
        elif model == 2: #SLM // EN
            h_invs = exp(D*xs_B)
            der_h = 1.0/(D*h_invs)
            der_h2 = -1.0/(D*h_invs**2)
            drift = k * (mu-h_invs)*h_invs
            noise2 = (D*h_invs)**2.0
        Ap = der_h*drift + 0.5*noise2*der_h2 #Drift of the transformed process in the OU bridge measure
        Ab = k_OU * (mu_OU - xs_B)           #Drift of the OU bridge measure


        #5) Girsanov exponent components (left-point rule)
        dx = np.diff(xs_B)

        state_integral = np.sum((Ap[:-1]-Ab[:-1]) * dx) #This could be computed exactly
        time_integral  = np.sum((Ap[:-1]**2 - Ab[:-1]**2) * dt) 

        log_weight = state_integral - 0.5 * time_integral
        weight = np.exp(log_weight)

        Ls[ii] = weight

    #6) Sample mean
    empirical_prop_transformed_process = np.mean(Ls)* prop_OU

    #7) Transforming propagator
    empirical_prop = empirical_prop_transformed_process*Jacobian

    #8) Optionally compute statistical errors
    std_error = np.std(Ls*prop_OU*Jacobian)/sqrt(NB)
    return empirical_prop, Ls, std_error

@njit
def prop_BCM_OU_bridges_optimal_params(x, t, y, s, params, model, NB, dt_B, errors = False):
    """
    Compute the propagator of model using Ornstein-Uhlenbeck (OU) bridges and Bridge change of measure technique.
    The (unconditioned) OU process is given by the SDE:
        dX_t = k_OU(mu_OU - X_t) dt + dW_t
    Parameters k_OU and mu_OU are chosen optimally by matching the transformed drift of the target process at the endpoints of the bridge.
    Available models:
    0 --> Ornstein Uhlenbeck                          dX_t= k(mu-X_t) dt + D dW_t
    1 --> Cox-Ingersoll-Ross // Birth death migration dX_t= k(mu-X_t) dt + D sqrt(X_t) dW_t
    2 --> Stochastic logistic                         dX_t= k(mu-X_t)X_t dt + D X_t dW_t
    """
    NB = int(NB)
    Ls = np.zeros(NB)
    mu_OU, k_OU = Match_OU_bridge_params(x, y, params, model)
    mu, k, D = params
    #1) Transform data. Same transformation as in prop_BCM_Wiener_bridges.
    if model == 0: #OU
        hx = x/D
        hy = y/D
        Jacobian = 1.0/D
    elif model == 1: #CIR // BDMM // DE
        hx = 2.0*sqrt(x)/D
        hy = 2.0*sqrt(y)/D
        Jacobian = 1.0/(D*sqrt(x))
    elif model == 2: #SLM
        hx = log(x)/D
        hy = log(y)/D
        Jacobian = 1.0/(D*x)
    else:
        print("ERROR: Model not recognized in prop_BCM_OU_bridges (Flag1)")
        return -1.0, np.zeros(NB), 0.0
    
    prop_OU = propagator_OU_process(hx, t-s, hy, mu_OU, k_OU, 1.0)  # OU propagator
    
    #2) Build the time grid for the Wiener bridges (only once)
    xs_B, ts_B = Setup_bridge_grid(tf = t-s, dt = dt_B)
    dt = np.diff(ts_B) 

    for ii in range(NB):
        #3) Simulate Wiener bridges
        ts_B, xs_B = Simulate_OU_bridge_path(ts_B, xs_B, tf = t-s, x0 = hy, xf = hx, mu = mu_OU, k = k_OU)
        #4) Compute drift of the transformed process
        if model == 0: #OU
            h_invs = D*xs_B
            der_h = np.full_like(h_invs, 1.0/D) #Due to njit, this cannot be a constant if in a different branch it is an array
            der_h2 = np.zeros_like(xs_B)
            drift = k * (mu-h_invs)
            noise2 = np.full_like(h_invs, D**2.0)
        elif model == 1: #CIR // BDMM // DE
            h_invs = (D*xs_B)**2.0/4.0
            der_h = 1.0/(D*sqrt(h_invs))
            der_h2 = -h_invs**(-1.5)/(2.0*D)
            drift = k * (mu-h_invs)
            noise2 = D**2.0 * h_invs
        elif model == 2: #SLM // EN
            h_invs = exp(D*xs_B)
            der_h = 1.0/(D*h_invs)
            der_h2 = -1.0/(D*h_invs**2)
            drift = k * (mu-h_invs)*h_invs
            noise2 = (D*h_invs)**2.0
        Ap = der_h*drift + 0.5*noise2*der_h2 #Drift of the transformed process in the OU bridge measure
        Ab = k_OU * (mu_OU - xs_B)           #Drift of the OU bridge measure


        #5) Girsanov exponent components (left-point rule)
        dx = np.diff(xs_B)

        state_integral = np.sum((Ap[:-1]-Ab[:-1]) * dx) #This could be computed exactly
        time_integral  = np.sum((Ap[:-1]**2 - Ab[:-1]**2) * dt) 

        log_weight = state_integral - 0.5 * time_integral
        weight = np.exp(log_weight)

        Ls[ii] = weight

    #6) Sample mean
    empirical_prop_transformed_process = np.mean(Ls)* prop_OU

    #7) Transforming propagator
    empirical_prop = empirical_prop_transformed_process*Jacobian

    #8) Optionally compute statistical errors
    std_error = np.std(Ls*prop_OU*Jacobian)/sqrt(NB)
    return empirical_prop, Ls, std_error

@njit
def prop_BCM_OU_bridges_optimal_params_trapezoid(x, t, y, s, params, model, NB, dt_B, errors = False):
    """
    Same as prop_BCM_OU_bridges_optimal_params, but using the trapezoid discretization (Sec. IV,
    Eqs. 30-35 of the theory notes) instead of the left-point rule. Since B=1 for both the
    transformed process and the OU bridge, the state (stochastic) integral of the Girsanov exponent
    is evaluated exactly through the primitive of Ap-Ab (Rp, Rb below), which depends only on the
    bridge endpoints hy, hx and is therefore computed once, before the bridge loop. Only the
    remaining time integral is discretized, now with the trapezoid rule instead of the left-point rule.
    """
    NB = int(NB)
    Ls = np.zeros(NB)
    mu_OU, k_OU = Match_OU_bridge_params(x, y, params, model)
    mu, k, D = params

    #1) Transform data. Same transformation as in prop_BCM_OU_bridges_optimal_params.
    if model == 0: #OU
        hx = x/D
        hy = y/D
        Jacobian = 1.0/D
    elif model == 1: #CIR // BDMM // DE
        hx = 2.0*sqrt(x)/D
        hy = 2.0*sqrt(y)/D
        Jacobian = 1.0/(D*sqrt(x))
    elif model == 2: #SLM
        hx = log(x)/D
        hy = log(y)/D
        Jacobian = 1.0/(D*x)
    else:
        print("ERROR: Model not recognized in prop_BCM_OU_bridges_optimal_params_trapezoid (Flag1)")
        return -1.0, np.zeros(NB), 0.0

    prop_OU = propagator_OU_process(hx, t-s, hy, mu_OU, k_OU, 1.0)  # OU propagator

    #2) Build the time grid for the OU bridges (only once)
    xs_B, ts_B = Setup_bridge_grid(tf = t-s, dt = dt_B)
    dt = np.diff(ts_B)

    #3) Exact state integral of the Girsanov exponent: int (Ap(y)-Ab(y)) dy = (Rp(hx)-Rp(hy)) - (Rb(hx)-Rb(hy)),
    #   with Rp the primitive of the transformed target drift Ap, and Rb the primitive of the OU bridge
    #   drift Ab. This only depends on the endpoints hy, hx, so it is the same for every bridge and is
    #   computed only once, here, instead of inside the loop.
    if model == 0: #OU
        Rp_hx = (k*mu/D)*hx - 0.5*k*hx**2.0
        Rp_hy = (k*mu/D)*hy - 0.5*k*hy**2.0
    elif model == 1: #CIR // BDMM // DE
        C = 2.0*k*mu/D**2.0 - 0.5
        Rp_hx = C*log(hx) - 0.25*k*hx**2.0
        Rp_hy = C*log(hy) - 0.25*k*hy**2.0
    elif model == 2: #SLM // EN
        Rp_hx = (k*mu/D - 0.5*D)*hx - (k/D**2.0)*exp(D*hx)
        Rp_hy = (k*mu/D - 0.5*D)*hy - (k/D**2.0)*exp(D*hy)

    Rb_hx = k_OU*mu_OU*hx - 0.5*k_OU*hx**2.0
    Rb_hy = k_OU*mu_OU*hy - 0.5*k_OU*hy**2.0

    exact_state_integral = (Rp_hx - Rb_hx) - (Rp_hy - Rb_hy)

    for ii in range(NB):
        #4) Simulate OU bridges
        ts_B, xs_B = Simulate_OU_bridge_path(ts_B, xs_B, tf = t-s, x0 = hy, xf = hx, mu = mu_OU, k = k_OU)

        #5) Trapezoid rule for the remaining time integral. Ap is the transformed target drift (same
        #   as in prop_BCM_OU_bridges_optimal_params) and Ap_deriv its derivative; Ab is the OU bridge
        #   drift, whose derivative is simply -k_OU.
        if model == 0: #OU
            Ap = k*mu/D - k*xs_B
            Ap_deriv = np.full_like(xs_B, -k) #Due to njit, this cannot be a constant if in a different branch it is an array
        elif model == 1: #CIR // BDMM // DE
            C = 2.0*k*mu/D**2.0 - 0.5
            Ap = C/xs_B - 0.5*k*xs_B
            Ap_deriv = -C/xs_B**2.0 - 0.5*k
        elif model == 2: #SLM // EN
            Ap = k*mu/D - 0.5*D - (k/D)*exp(D*xs_B)
            Ap_deriv = -k*exp(D*xs_B)
        Ab = k_OU * (mu_OU - xs_B)
        q = Ap**2.0 - Ab**2.0 + Ap_deriv + k_OU #r' = Ap_deriv - Ab_deriv = Ap_deriv + k_OU

        time_integral = np.sum((q[:-1] + q[1:]) * dt) / 2.0 #Trapezoid rule

        log_weight = exact_state_integral - 0.5 * time_integral
        weight = np.exp(log_weight)

        Ls[ii] = weight

    #6) Sample mean
    empirical_prop_transformed_process = np.mean(Ls)* prop_OU

    #7) Transforming propagator
    empirical_prop = empirical_prop_transformed_process*Jacobian

    #8) Optionally compute statistical errors
    std_error = np.std(Ls*prop_OU*Jacobian)/sqrt(NB)
    return empirical_prop, Ls, std_error

@njit
def Transformed_drift(y_val, params, model):
    """Evaluate the transformed target drift A'(y) (see prop_BCM_OU_bridges, step 4) at a single point y_val."""
    mu, k, D = params
    if model == 0: #OU
        h_inv = D*y_val
        der_h = 1.0/D
        der_h2 = 0.0
        drift = k*(mu-h_inv)
        noise2 = D**2.0
    elif model == 1: #CIR // BDMM // DE
        h_inv = (D*y_val)**2.0/4.0
        der_h = 1.0/(D*sqrt(h_inv))
        der_h2 = -h_inv**(-1.5)/(2.0*D)
        drift = k*(mu-h_inv)
        noise2 = D**2.0 * h_inv
    elif model == 2: #SLM // EN
        h_inv = exp(D*y_val)
        der_h = 1.0/(D*h_inv)
        der_h2 = -1.0/(D*h_inv**2)
        drift = k*(mu-h_inv)*h_inv
        noise2 = (D*h_inv)**2.0
    else:
        print("ERROR: Model not recognized in Transformed_drift (Flag1)")
        return 0.0
    return der_h*drift + 0.5*noise2*der_h2

@njit
def Match_OU_bridge_params(x, y, params, model):
    """
    Choose (mu_OU, k_OU) for the OU auxiliary bridge by matching its drift
    a(z) = k_OU*(mu_OU - z) to the target's transformed drift A'(z) exactly
    at the two bridge endpoints hy = f(y), hx = f(x) (secant-line matching).
    This is the zeroth-order solution to requiring that both the process
    and the time Girsanov integrals vanish (endpoint/trapezoidal quadrature).

    Returns
    -------
    params_B : tuple
        (mu_OU, k_OU), ready to pass to prop_BCM_OU_bridges.
    """
    if model == 0: #OU
        _, _, D = params
        hx = x/D
        hy = y/D
    elif model == 1: #CIR // BDMM // DE
        _, _, D = params
        hx = 2.0*sqrt(x)/D
        hy = 2.0*sqrt(y)/D
    elif model == 2: #SLM
        _, _, D = params
        hx = log(x)/D
        hy = log(y)/D
    else:
        print("ERROR: Model not recognized in Match_OU_bridge_params (Flag1)")
        return 0.0, 0.0

    if hx == hy:
        raise ValueError("x and y map to the same point; cannot fit a secant (choose x != y).")

    Ap_hy = Transformed_drift(hy, params, model)
    Ap_hx = Transformed_drift(hx, params, model)

    k_OU = (Ap_hy - Ap_hx) / (hx - hy)
    mu_OU = hy + Ap_hy/k_OU

    return mu_OU, k_OU

@njit
def prop_BCM_Wiener_bridges(x, t, y, s, params, model, NB, dt_B, errors = False):
    """
    Compute the propagator of model using Wiener bridges and Bridge change of measure technique.
    Available models:
    0 --> Ornstein Uhlenbeck                          dX_t= k(mu-X_t) dt + D dW_t
    1 --> Cox-Ingersoll-Ross // Birth death migration dX_t= k(mu-X_t) dt + D sqrt(X_t) dW_t
    2 --> Stochastic logistic                         dX_t= k(mu-X_t)X_t dt + D X_t dW_t
    """
    NB = int(NB)
    Ls = np.zeros(NB)

    #1) Transform data
    if model == 0: #OU
        mu, k, D = params
        hx = x/D
        hy = y/D
        Jacobian = 1.0/D
    elif model == 1: #CIR // BDMM // DE
        mu, k, D = params
        hx = 2.0*sqrt(x)/D
        hy = 2.0*sqrt(y)/D
        Jacobian = 1.0/(D*sqrt(x))
    elif model == 2: #SLM
        mu, k, D = params
        hx = log(x)/D
        hy = log(y)/D
        Jacobian = 1.0/(D*x)
    else:
        print("ERROR: Model not recognized in prop_BCM_Wiener_bridges (Flag1)")
        return -1.0, np.zeros(NB), 0.0
    
    prop_WI = propagator_WI_process(hx, t-s, hy, 1.0)  # Wiener propagator

    #2) Build the time grid for the Wiener bridges (only once)
    xs_B, ts_B = Setup_bridge_grid(tf = t-s, dt = dt_B)
    dt = np.diff(ts_B) 

    for ii in range(NB):
        #3) Simulate Wiener bridges
        ts_B, xs_B = Simulate_Wiener_bridge_path(ts_B, xs_B, tf = t-s, x0 = hy, xf = hx)
        #4) Compute drift of the transformed process
        if model == 0: #OU
            h_invs = D*xs_B
            der_h = np.full_like(h_invs, 1.0/D) #Due to njit, this cannot be a constant if in a different branch it is an array
            der_h2 = np.zeros_like(xs_B)
            drift = k * (mu-h_invs)
            noise2 = np.full_like(h_invs, D**2.0)
        elif model == 1: #CIR // BDMM // DE
            h_invs = (D*xs_B)**2.0/4.0
            der_h = 1.0/(D*sqrt(h_invs))
            der_h2 = -h_invs**(-1.5)/(2.0*D)
            drift = k * (mu-h_invs)
            noise2 = D**2.0 * h_invs
        elif model == 2: #SLM // EN
            h_invs = exp(D*xs_B)
            der_h = 1.0/(D*h_invs)
            der_h2 = -1.0/(D*h_invs**2)
            drift = k * (mu-h_invs)*h_invs
            noise2 = (D*h_invs)**2.0
        Ap = der_h*drift + 0.5*noise2*der_h2


        #5) Girsanov exponent components (left-point rule)
        dx = np.diff(xs_B)

        state_integral = np.sum(Ap[:-1] * dx) #This could be computed exactly
        time_integral  = np.sum(Ap[:-1]**2 * dt) 

        log_weight = state_integral - 0.5 * time_integral
        weight = np.exp(log_weight)

        Ls[ii] = weight

    #6) Sample mean
    empirical_prop_transformed_process = np.mean(Ls)* prop_WI

    #7) Transforming propagator
    empirical_prop = empirical_prop_transformed_process*Jacobian

    #8) Optionally compute statistical errors
    std_error = np.std(Ls*prop_WI*Jacobian)/sqrt(NB)
    return empirical_prop, Ls, std_error


@njit
def _seed_numba_rng(seed):
    np.random.seed(seed)


def set_seed(seed):
    """Seed both NumPy's and Numba's random generators (they are independent)."""
    np.random.seed(seed)
    _seed_numba_rng(seed)

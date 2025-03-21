# v_currents.py
import numpy as np

def ina(hode, v, frt, xh, xj, xm, xnai, xnao):
    """
    Fast sodium current - exactly matches Fortran subroutine
    
    Returns:
    --------
    xina : float
        Sodium current value
    xh : float
        h-gate value (updated)
    xj : float
        j-gate value (updated)
    xm : float 
        m-gate value (updated)
    """
    # Parameters directly from Fortran
    gna = 6.0
    XKMCAM = 0.3
    deltax = -0.18
    
    # Sodium reversal potential
    ena = (1.0/frt) * np.log(xnao/xnai)
    
    # Activation gate parameters
    am = 0.32 * (v + 47.13) / (1.0 - np.exp(-0.1 * (v + 47.13)))
    bm = 0.08 * np.exp(-v/11.0)
    
    # Ca-calmodulin factor (disabled in Fortran)
    camfact = 0.0
    vshift = 0.0
    
    # Voltage with shift
    vx = v - vshift
    
    # Inactivation gates (h and j)
    if vx < -40.0:
        ah = 0.135 * np.exp((80.0 + vx) / -6.8)
        bh = 3.56 * np.exp(0.079 * vx) + 310000.0 * np.exp(0.35 * vx)
        
        aj1a = -127140.0 * np.exp(0.2444 * vx)
        aj1b = 0.00003474 * np.exp(-0.04391 * vx)
        aj1c = (vx + 37.78) / (1.0 + np.exp(0.311 * (vx + 79.23)))
        
        aj = (1.0 + camfact * deltax) * (aj1a - aj1b) * aj1c
        bj = (0.1212 * np.exp(-0.01052 * vx)) / (1.0 + np.exp(-0.1378 * (vx + 40.14)))
    else:
        ah = 0.0
        bh = 1.0 / (0.13 * (1.0 + np.exp((vx + 10.66) / -11.1)))
        aj = 0.0
        
        bja = 0.3 * np.exp(-0.0000002535 * vx)
        bjb = 1.0 + np.exp(-0.1 * (vx + 32.0))
        bj = bja / bjb
    
    # Time constants
    tauh = 1.0 / (ah + bh)
    tauj = 1.0 / (aj + bj)
    taum = 1.0 / (am + bm)
    
    # Current calculation
    xina = gna * xh * xj * xm * xm * xm * (v - ena)
    
    # State updates - matches Fortran equation form exactly
    xh = ah / (ah + bh) - ((ah / (ah + bh)) - xh) * np.exp(-hode / tauh)
    xj = aj / (aj + bj) - ((aj / (aj + bj)) - xj) * np.exp(-hode / tauj)
    xm = am / (am + bm) - ((am / (am + bm)) - xm) * np.exp(-hode / taum)
    
    return xina, xh, xj, xm

def ikr(hode, v, frt, xko, xki, xr):
    """
    Rapid delayed rectifier potassium current - exactly matches Fortran subroutine
    """
    # K reversal potential
    ek = (1.0/frt) * np.log(xko/xki)
    
    # Conductance scaling with Ko
    gss = np.sqrt(xko/5.40)
    
    # Activation kinetics
    xkrv1 = 0.00138 * (v + 7.0) / (1.0 - np.exp(-0.123 * (v + 7.0)))
    xkrv2 = 0.00061 * (v + 10.0) / (np.exp(0.145 * (v + 10.0)) - 1.0)
    taukr = 1.0 / (xkrv1 + xkrv2)
    
    # Steady-state activation
    xkrinf = 1.0 / (1.0 + np.exp(-(v + 50.0) / 7.5))
    
    # Inward rectification
    rg = 1.0 / (1.0 + np.exp((v + 33.0) / 22.4))
    
    # Current calculation
    gkr = 0.007836  # IKr conductance
    xikr = gkr * gss * xr * rg * (v - ek)
    
    # State update
    xr = xkrinf - (xkrinf - xr) * np.exp(-hode / taukr)
    
    return xikr, xr

def iks(hode, v, frt, ci, xnao, xnai, xko, xki, xs1, qks):
    """
    Slow delayed rectifier potassium current - exactly matches Fortran subroutine
    """
    # Parameters
    prnak = 0.01833
    
    # Calcium dependence
    qks_inf = 0.6 * ci
    tauqks = 1000.0
    
    # Reversal potential
    eks = (1.0/frt) * np.log((xko + prnak * xnao) / (xki + prnak * xnai))
    
    # Activation gate
    xs1ss = 1.0 / (1.0 + np.exp(-(v - 1.50) / 16.70))
    xs2ss = xs1ss  # Not used directly
    
    # Time constant
    tauxs = 1.0 / (0.0000719 * (v + 30.0) / (1.0 - np.exp(-0.148 * (v + 30.0))) + 
                  0.000131 * (v + 30.0) / (np.exp(0.0687 * (v + 30.0)) - 1.0))
    
    # Conductance and current
    gksx = 0.200 * 0.7  # IKs conductance
    xiks = gksx * qks * xs1**2 * (v - eks)
    
    # State updates
    xs1 = xs1ss - (xs1ss - xs1) * np.exp(-hode / tauxs)
    qks = qks + hode * (qks_inf - qks) / tauqks
    
    return xiks, xs1, qks

def ik1(v, frt, xki, xko):
    """
    Inward rectifier potassium current - exactly matches Fortran subroutine
    """
    # Reversal potential
    ek = (1.0/frt) * np.log(xko/xki)
    
    # Conductance
    gkix = 0.60 * 1.0  # IK1 conductance (reduced in Grandi model)
    gki = gkix * (np.sqrt(xko/5.4))
    
    # Inward rectification
    aki = 1.02 / (1.0 + np.exp(0.2385 * (v - ek - 59.215)))
    bki = (0.49124 * np.exp(0.08032 * (v - ek + 5.476)) + 
          np.exp(0.06175 * (v - ek - 594.31))) / (1.0 + np.exp(-0.5143 * (v - ek + 4.753)))
    
    # Open probability and current
    xkin = aki / (aki + bki)
    xik1 = gki * xkin * (v - ek)
    
    return xik1

def ito(hode, v, frt, xki, xko, xtof, ytof, xtos, ytos, gtof, gtos):
    """
    Transient outward current - exactly matches Fortran subroutine
    """
    # Reversal potential
    ek = (1.0/frt) * np.log(xko/xki)
    
    # Common parameters
    rt1 = -(v + 3.0) / 15.0
    rt2 = (v + 33.5) / 10.0
    rt3 = (v + 60.0) / 10.0
    
    # Slow component
    xtos_inf = 1.0 / (1.0 + np.exp(rt1))
    ytos_inf = 1.0 / (1.0 + np.exp(rt2))
    
    rs_inf = 1.0 / (1.0 + np.exp(rt2))
    
    txs = 9.0 / (1.0 + np.exp(-rt1)) + 0.5
    tys = 3000.0 / (1.0 + np.exp(rt3)) + 30.0
    
    # Slow Ito current
    xitos = gtos * xtos * (ytos + 0.5 * rs_inf) * (v - ek)
    
    # Slow component state updates
    xtos = xtos_inf - (xtos_inf - xtos) * np.exp(-hode / txs)
    ytos = ytos_inf - (ytos_inf - ytos) * np.exp(-hode / tys)
    
    # Fast component (Shannon et al. 2005)
    xtof_inf = xtos_inf
    ytof_inf = ytos_inf
    
    rt4 = -(v / 30.0) * (v / 30.0)
    rt5 = (v + 33.5) / 10.0
    txf = 3.5 * np.exp(rt4) + 1.5
    tyf = 20.0 / (1.0 + np.exp(rt5)) + 20.0
    
    # Fast Ito current
    xitof = gtof * xtof * ytof * (v - ek)
    
    # Fast component state updates
    xtof = xtof_inf - (xtof_inf - xtof) * np.exp(-hode / txf)
    ytof = ytof_inf - (ytof_inf - ytof) * np.exp(-hode / tyf)
    
    # Total Ito current
    xito = xitos + xitof
    
    return xito, xtof, ytof, xtos, ytos

def inak(v, frt, xko, xnao, xnai):
    """
    Na-K pump current - exactly matches Fortran subroutine
    """
    # Parameters
    xibarnak = 1.50
    xkmko = 1.5    # Ko half-saturation constant
    xkmnai = 12.0  # Nai half-saturation constant
    hh = 1.0       # Na dependence exponent
    
    # Voltage dependence
    sigma = (np.exp(xnao / 67.3) - 1.0) / 7.0
    fnak = 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * v * frt) + 0.0365 * sigma * np.exp(-v * frt))
    
    # Calculate pump current
    xinak = xibarnak * fnak * (1.0 / (1.0 + (xkmnai / xnai)**hh)) * xko / (xko + xkmko)
    
    return xinak
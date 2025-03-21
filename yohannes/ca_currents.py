# ca_currents.py
import numpy as np
import random

def binom(n, p):
    """
    Generate a binomial random number - mimics Fortran's implementation
    
    Parameters:
    -----------
    n : int
        Number of trials
    p : float
        Probability of success
    
    Returns:
    --------
    m : int
        Number of successes
    """
    if n <= 0 or p <= 0:
        return 0
        
    xlog = -np.log(1.0 - p)
    xsum = 0.0
    i = 1
    
    while True:
        # Check if we've reached the limit of trials
        if i > n:
            return n  # All trials were successful
            
        r1 = random.random()
        xx2 = float(n - i + 1)
        
        # Safety check to avoid division by zero
        if xx2 <= 0:
            return i - 1
            
        hx = -np.log(r1) / xx2
        xsum = xsum + hx
        
        if xsum > xlog:
            break
            
        i += 1
    
    m = i - 1
    return m

def binevol(nt, nx, alpha, beta, dt):
    """
    Binomial evolution for spark generation - mimics Fortran implementation
    
    Parameters:
    -----------
    nt : int
        Total number of dyads
    nx : int
        Current number of sparking dyads
    alpha : float
        Spark activation rate
    beta : float
        Spark termination rate
    dt : float
        Time step
    
    Returns:
    --------
    ndeltap : int
        Number of new sparks
    ndeltam : int
        Number of terminated sparks
    """
    # Number of available (non-sparking) clusters
    na = nt - nx
    
    # Calculate new sparks (activation)
    if na > 0:
        xrate = alpha * dt  # Activation probability
        ndeltap = binom(na, xrate)
    else:
        ndeltap = 0
    
    # Calculate terminated sparks (inactivation)
    if nx > 0:
        xrate = beta * dt  # Inactivation probability
        ndeltam = binom(nx, xrate)
    else:
        ndeltam = 0
    
    return ndeltap, ndeltam

def uptake(ci, vup):
    """
    Calculate SR calcium uptake rate - direct from Fortran
    
    Parameters:
    -----------
    ci : float
        Internal calcium concentration
    vup : float
        Maximum uptake rate
    
    Returns:
    --------
    xup : float
        Actual uptake rate
    """
    # Constants from Fortran
    Ki = 0.30
    Knsr = 800.0  # Not used in Fortran code but defined
    HH = 3.00
    
    # Hill equation for uptake
    xup = vup * ci**HH / (Ki**HH + ci**HH)
    
    return xup

def total(ci):
    """
    Convert free calcium to total calcium - direct from Fortran
    
    Parameters:
    -----------
    ci : float
        Free calcium concentration
    
    Returns:
    --------
    cit : float
        Total calcium concentration
    """
    # Buffer parameters from Fortran
    bcal = 24.0
    xkcal = 7.0
    srmax = 47.0
    srkd = 0.6
    
    # Calculate buffered calcium
    bix = bcal * ci / (xkcal + ci)   # Calmodulin buffering
    six = srmax * ci / (srkd + ci)    # SR buffering
    
    # Total calcium
    cit = ci + bix + six
    
    return cit

def xfree(cit):
    """
    Convert total calcium to free calcium - direct from Fortran
    
    Parameters:
    -----------
    cit : float
        Total calcium concentration
    
    Returns:
    --------
    ci : float
        Free calcium concentration
    """
    # Constants from Fortran
    a = 2.23895
    b = 52.0344
    c = 0.666509
    
    y = cit
    
    # Quadratic formula from Fortran
    xa = (b + a*c - y)**2 + 4.0*a*c*y
    ci = (-b - a*c + y + np.sqrt(xa)) / (2.0*a)
    
    return ci

def inaca(v, frt, xnai, xnao, cao, ci):
    """
    Na-Ca exchanger current - direct from Fortran
    
    Parameters:
    -----------
    v : float
        Membrane potential
    frt : float
        F/RT constant
    xnai : float
        Internal sodium concentration
    xnao : float
        External sodium concentration
    cao : float
        External calcium concentration
    ci : float
        Internal calcium concentration
    
    Returns:
    --------
    xinacaq : float
        NCX current
    """
    # Convert to millimolar
    cim = ci / 1000.0
    
    # Calculate voltage-dependent terms
    zw3a = xnai**3 * cao * np.exp(v * 0.35 * frt)
    zw3b = xnao**3 * cim * np.exp(v * (0.35 - 1.0) * frt)
    
    zw3 = zw3a - zw3b
    zw4 = 1.0 + 0.2 * np.exp(v * (0.35 - 1.0) * frt)
    
    # Calculate Ca-dependent factor
    xkdna = 0.3  # micro M
    aloss = 1.0 / (1.0 + (xkdna / ci)**3)
    
    # Half-saturation constants (all in mM)
    xmcao = 1.3
    xmnao = 87.5
    xmnai = 12.3
    xmcai = 0.0036
    
    # Calculate denominator terms
    yz1 = xmcao * xnai**3 + xmnao**3 * cim
    yz2 = xmnai**3 * cao * (1.0 + cim / xmcai)
    yz3 = xmcai * xnao**3 * (1.0 + (xnai / xmnai)**3)
    yz4 = xnai**3 * cao + xnao**3 * cim
    
    zw8 = yz1 + yz2 + yz3 + yz4
    
    # Calculate NCX current
    xinacaq = aloss * zw3 / (zw4 * zw8)
    
    return xinacaq

def ica(v, frt, cao, ci, pox):
    """
    L-type calcium current - direct from Fortran
    
    Parameters:
    -----------
    v : float
        Membrane potential
    frt : float
        F/RT constant
    cao : float
        External calcium concentration
    ci : float
        Internal calcium concentration
    pox : float
        Open probability of L-type calcium channels
    
    Returns:
    --------
    rca : float
        Actual calcium flux
    xicaq : float
        ICa current
    """
    # Constants
    xf = 96.485  # Faraday's constant
    pca = 0.00054  # Permeability constant from Luo-Rudy
    
    # Calculate electrochemical driving force
    za = v * 2.0 * frt
    
    factor1 = 4.0 * pca * xf * frt
    factor = v * factor1
    
    # Convert to millimolar
    cim = ci / 1000.0
    
    # Compute driving force with special case for small voltage
    if abs(za) < 0.001:
        rca = factor1 * (cim * np.exp(za) - 0.341 * cao) / (2.0 * frt)
    else:
        rca = factor * (cim * np.exp(za) - 0.341 * cao) / (np.exp(za) - 1.0)
    
    # Calculate current
    xicaq = rca * pox
    
    return rca, xicaq

def markov(hode, v, ci, c1, c2, xi1, xi2, po, c1s, c2s, xi1s, xi2s, pos, alpha, bts, zxr):
    """
    Markov model for L-type calcium channel gating - direct from Fortran
    
    Parameters:
    -----------
    Multiple state variables - see Fortran code
    
    Returns:
    --------
    Updated state variables
    """
    # Ca-independent rates
    a23 = 0.3
    a32 = 3.0
    a42 = 0.00224
    
    # Voltage-dependent activation
    vth = 1.0
    s6 = 7.0
    
    poinf = 1.0 / (1.0 + np.exp(-(v - vth) / s6))
    taupo = 1.0
    
    a12 = poinf / taupo
    a21 = (1.0 - poinf) / taupo
    
    # Voltage-dependent inactivation
    vy = -40.0
    sy = 4.0
    prv = 1.0 - 1.0 / (1.0 + np.exp(-(v - vy) / sy))
    
    vyr = -40.0
    syr = 10.0
    
    recovx = 10.0 + 4954.0 * np.exp(v / 15.6)
    recov = 1.5 * recovx
    
    tauba = (recov - 450.0) * prv + 450.0
    tauba = tauba / 2.0
    
    poix = 1.0 / (1.0 + np.exp(-(v - vyr) / syr))
    
    a15 = poix / tauba
    a51 = (1.0 - poix) / tauba
    
    vx = -40.0
    sx = 3.0
    tau3 = 3.0
    poi = 1.0 / (1.0 + np.exp(-(v - vx) / sx))
    a45 = (1.0 - poi) / tau3
    
    # Ca-dependent rates
    cat = 0.8
    
    fca = 1.0 / (1.0 + (cat / ci)**2)
    a24 = 0.00413 + zxr * fca
    a34 = 0.00195 + zxr * fca
    
    a43 = a34 * (a23 / a32) * (a42 / a24)
    a54 = a45 * (a51 / a15) * (a24 / a42) * (a12 / a21)
    
    fcax = 1.0
    a24s = 0.00413 + zxr * fcax
    a34s = 0.00195 + zxr * fcax
    
    a43s = a34s * (a23 / a32) * (a42 / a24s)
    a54s = a45 * (a51 / a15) * (a24s / a42) * (a12 / a21)
    
    # State dynamics
    dpo = a23 * c1 + a43 * xi1 - (a34 + a32) * po - alpha * po + bts * pos
    dc2 = a21 * c1 + a51 * xi2 - (a15 + a12) * c2 + bts * c2s
    dc1 = a12 * c2 + a42 * xi1 + a32 * po - (a21 + a23 + a24) * c1 + bts * c1s
    dxi1 = a24 * c1 + a54 * xi2 + a34 * po - (a45 + a42 + a43) * xi1 + bts * xi1s
    
    dpos = a23 * c1s + a43s * xi1s - (a34s + a32) * pos + alpha * po - bts * pos
    dc2s = a21 * c1s + a51 * xi2s - (a15 + a12) * c2s - bts * c2s
    dc1s = a12 * c2s + a42 * xi1s + a32 * pos - (a21 + a23 + a24s) * c1s - bts * c1s
    dxi1s = a24s * c1s + a54s * xi2s + a34s * pos - (a45 + a42 + a43s) * xi1s - bts * xi1s
    dxi2s = a45 * xi1s + a15 * c2s - (a51 + a54s) * xi2s - bts * xi2s
    
    # Update states
    po = po + dpo * hode
    c1 = c1 + dc1 * hode
    c2 = c2 + dc2 * hode
    xi1 = xi1 + dxi1 * hode
    
    pos = pos + dpos * hode
    c1s = c1s + dc1s * hode
    c2s = c2s + dc2s * hode
    xi1s = xi1s + dxi1s * hode
    xi2s = xi2s + dxi2s * hode
    
    # Conservation of probability
    xi2 = 1.0 - c1 - c2 - po - xi1 - pos - c1s - c2s - xi1s - xi2s
    
    return po, c1, c2, xi1, xi2, pos, c1s, c2s, xi1s, xi2s
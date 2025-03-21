# main_model.py

import numpy as np
from numba import njit 
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Import from your local modules
from v_currents import (
    ina, ikr, iks, ik1, ito, inak
)
from ca_currents import (
    binom, binevol, uptake, total, xfree, inaca, ica, markov
)

njit()
def main():
    # === Simulation parameters ===
    nstim = 200       # number of beats - FIXED: changed from 24 to 80 to match Fortran
    rbcl = 600.0     # pacing cycle length in ms - FIXED: changed from 500.0 to 400.0
    dt = 0.1
  
    # === Model parameters from Fortran ===
    gicai = 2.20
    gtos  = 0.04
    gtof  = 0.15
    gnacai= 1.5
    zxr   = 0.09
    nbt   = 4000    # FIXED: changed from 4000 to 10000
    cxinit= 1200.0   # FIXED: changed from 1000.0 to 1200.0

    # Sodium concentration logic
    xmx   = -2.0/250.0
    xnai  = xmx*520.0 + 16.0

    # Constants
    xnao  = 136.0
    xki   = 140.0
    xko   = 5.40
    cao   = 1.8
    temp  = 308.0
    xxr   = 8.314
    xf    = 96.485
    frt   = xf/(xxr*temp)

    # === Initial conditions ===
    cb   = 0.1
    ci   = 0.1
    csrb = cxinit
    csri = cxinit
    po   = 0.0
    c1   = 0.0
    c2   = 1.0
    xi1  = 0.0
    xi2  = 0.0

    pos  = 0.0
    c1s  = 0.0
    c2s  = 0.0
    xi1s = 0.0
    xi2s = 0.0

    # total() in Fortran
    cit = total(ci)
    cbt = total(cb)

    nsb  = 5
    v    = -90.0
    xm   = 0.001
    xh   = 1.0
    xj   = 1.0
    xr   = 0.0
    xs1  = 0.3
    qks  = 0.2
    xtos = 0.01
    ytos = 0.9
    xtof = 0.02
    ytof = 0.8

    dv   = 0.0  # track slope for adaptive substep
    t    = 0.0

    # Data arrays (replacing the Fortran 'OPEN' file writes)
    time_list = []
    v_list    = []
    cb_list   = []
    csrb_list = []


    # === Main integration ===
    for iz in tqdm(range(nstim)):
        nstep = int(rbcl / dt)
        for ncount in range(nstep):
            time = float(ncount) * dt

            # Use dv from previous iteration for sub-step
            adq = abs(dv)
            if adq > 25.0:
                mstp = 10
            else:
                mstp = 1

            hode = dt / float(mstp)

            # 1) Convert total Ca to free Ca
            ci = xfree(cit)
            cb = xfree(cbt)

            # fraction of clusters with sparks
            pb = float(nsb) / float(nbt)

            # uptake at boundary, interior
            vupb = 0.4
            vupi = 0.4
            xupb = uptake(cb, vupb)
            xupi = uptake(ci, vupi)

            # inaca & iCa
            xinaca1 = inaca(v, frt, xnai, xnao, cao, cb)
            xinacaq = gnacai * xinaca1

            pox = po + pos
            # Using exact Fortran implementation
            rca, xicaq = ica(v, frt, cao, cb, pox)
            xicaq = gicai * 130.0 * xicaq  # Using the actual xicaq from ica function

            # Spark rate at dyads
            qq    = 0.5
            ab    = 35.0 * qq
            csrx  = 600.0
            phisr = 1.0/(1.0 + (csrx / csrb) ** 10)
            # FIXED: Use rca instead of rcaq for alphab calculation
            alphab= ab * abs(rca) * po * phisr
            bts   = 1.0/30.0

            # Markov gating - Using exact Fortran implementation
            markov_result = markov(hode, v, cb, c1, c2, xi1, xi2, po,
                                  c1s, c2s, xi1s, xi2s, pos,
                                  alphab, bts, zxr)
            
            # Unpack the results
            po, c1, c2, xi1, xi2, pos, c1s, c2s, xi1s, xi2s = markov_result

            # RyR current at boundary
            gsrb = (0.01 / 1.5) * 1.0
            xryrb = gsrb * csrb * pb

            # Spark rate in interior (inactive => xryri=0)
            xryri = 0.0

            # Volume compartments & flux
            vi   = 0.50
            vb   = 1.0
            vbi   = vb / vi
            vbisr = vbi
            vq    = 30.0
            visr  = 30.0
            vbsr  = vq
          
            tau1  = 5.0
            tau2  = 5.0

            dfbi   = (cb - ci) / tau1
            dfbisr = (csrb - csri) / tau2

            # Net boundary current from LCC + NCX
            xsarc  = -xicaq + xinacaq

            dcbt  = xryrb - xupb + xsarc - dfbi
            dcsrb = vbsr * (-xryrb + xupb) - dfbisr

            dcit  = xryri - xupi + vbi * dfbi
            dcsri = visr * (-xryri + xupi) + vbisr * dfbisr

            cbt  += dcbt  * dt
            cit  += dcit * dt
            csrb += dcsrb * dt
            csri += dcsri * dt

            # Binomial evolution of sparks - Using exact Fortran implementation
            nsbx = nsb
            ndeltapx, ndeltamx = binevol(nbt, nsbx, alphab, bts, dt)
            
            if (ndeltamx > nsbx) or (ndeltapx > nbt):
                nsb = 0
            else:
                nsb = nsb + ndeltapx - ndeltamx

            # 2) Sub-step loop for voltage ODE
            for _ in range(mstp):
                wca   = 12.0
                xinaca = wca * xinacaq
                xica   = 2.0 * wca * xicaq

                # Voltage-dependent currents (Ina, IKr, IKs, IK1, Ito, INaK)
                # Using the exact Fortran implementation

                # ina
                xina, xh, xj, xm = ina(hode, v, frt, xh, xj, xm, xnai, xnao)

                # ikr
                xikr, xr = ikr(hode, v, frt, xko, xki, xr)

                # iks
                xiks, xs1, qks = iks(hode, v, frt, cb, xnao, xnai, xko, xki, xs1, qks)

                # ik1
                xik1 = ik1(v, frt, xki, xko)

                # ito
                xito, xtof, ytof, xtos, ytos = ito(hode, v, frt, xki, xko, xtof, ytof,
                                                  xtos, ytos, gtof, gtos)

                # inak
                xinak = inak(v, frt, xko, xnao, xnai)

                # Stim for first 1 ms each beat
                if time < 1.0:
                    stim = 80.0
                else:
                    stim = 0.0

                dvh = -(xina + xik1 + xikr + xiks + xito + xinaca + xica + xinak) + stim
                v = v + dvh * hode
                dv = dvh  # track slope for next iteration

            # Store data
            time_list.append(t)
            v_list.append(v)
            cb_list.append(cb)
            csrb_list.append(csrb)

            # Increment time
            t += dt

    # Write final arrays or plot
    plt.figure(figsize=(12, 8))
    
    # Plot membrane potential
    plt.subplot(3, 1, 1)
    plt.plot(time_list, v_list)
    plt.ylabel("Voltage (mV)")
    plt.title("Simulated Single Cell Action Potential")
    plt.xlim(0,10000)
    
    # Plot boundary calcium
    plt.subplot(3, 1, 2)
    plt.plot(time_list, cb_list)
    plt.ylabel("CB (μM)")
    plt.xlim(0,10000)

    
    # Plot SR calcium
    plt.subplot(3, 1, 3)
    plt.plot(time_list, csrb_list)
    plt.ylabel("CSRB (μM)")
    plt.xlabel("Time (ms)")
    plt.xlim(0,10000)

    
    plt.tight_layout()
    plt.show()

    # Also save data to files to match Fortran's behavior
    # np.savetxt('vx.dat', np.column_stack((time_list, v_list)))
    # np.savetxt('cbx.dat', np.column_stack((time_list, cb_list)))
    # np.savetxt('cjx.dat', np.column_stack((time_list, csrb_list)))
   
if __name__ == "__main__":
    main()
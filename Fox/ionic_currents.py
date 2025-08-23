import numpy as np
from numba import njit
from parameters import *

# VERIFIED
njit()
def INa(t,var,df):
    V = var[0]
    m = var[5]
    h = var[6]    
    j = var[7]

    # Constants:
    ENa = R*T/F*np.log(Nao/Nai)
    alpham = 0.32 * (V + 47.13) / (1 - np.exp(-0.1 * (V + 47.13)))
    betam = 0.08 * np.exp(-V/11)
    alphah = 0.135 * np.exp((V + 80)/-6.8)
    betah = 7.5/(1 + np.exp(-0.1*(V + 11)))
    alphaj = (0.175 * np.exp((V + 100)/-23)) / (1 + np.exp(0.15 * (V + 79)))
    betaj = 0.3/(1 + np.exp(-0.1*(V + 32)))

    # Remember, voltage is added upon
    INa = GNa * (m**3) * h * j * (V - ENa)
    df[0] += INa
    df[5] = (alpham*(1-m)) - betam*m
    df[6] = (alphah*(1-h)) - betah*h
    df[7] = (alphaj*(1-j)) - betaj*j

    return df, ENa

# VERIFIED
njit()
def IK1(t,var,df):
    V = var[0]

    # Constants:
    EK = R*T/F*np.log(Ko/Ki)
    K1inf = 1/(2+np.exp(1.62*F/(R*T)*(V-EK)))
    
    IK1 = GK1 * K1inf * Ko / (Ko + KmK1) * (V - EK)
    df[0] += IK1

    return df, EK

# VERIFIED
njit()
def IKr(t,var,df,EK):
    V = var[0]
    XKr = var[9]

    # Constants:
    RV = 1/(1 + 2.5*np.exp(0.1*(V + 28)))
    tauKr = 43 + (1 / (np.exp(-5.495 + 0.1691*V) + np.exp(-7.677 - 0.0128*V)))
    XKrinf = 1 / (1 + np.exp(-2.182 - 0.1819*V))

    IKr = GKr * RV * XKr * np.sqrt(Ko/4) * (V - EK)
    df[0] += IKr
    df[9] = (XKrinf - XKr) / tauKr

    return df

# VERIFIED
njit()
def IKs(t,var,df):
    V = var[0]
    XKs = var[10]

    # Constants:
    EKs = R*T/F*np.log((Ko + 0.01833*Nao)/(Ki + 0.01833*Nai))
    XKsinf = 1 / (1 + np.exp((V-16)/-13.6))
    Vminus10 = V-10
    tauKs = (0.0000719*Vminus10/(1 - np.exp(-0.148*Vminus10))) 
    tauKs += (0.000131*Vminus10/(np.exp(0.0687*Vminus10) - 1))
    tauKs = 1/tauKs

    IKs = GKs * XKs**2 * (V - EKs)
    df[0] += IKs
    df[10] = (XKsinf - XKs)/tauKs

    return df

# VERIFIED
njit()
def Ito(t,var,df,EK):
    V = var[0]
    Xto = var[11]
    Yto = var[12]

    # Constants:
    V_modified = (V+33.5)/5
    alphaXto = 0.04516*np.exp(0.03577*V)
    betaXto = 0.0989*np.exp(-0.06237*V)
    alphaYto = 0.005415*np.exp(-V_modified)/(1 + 0.051335*np.exp(-V_modified))
    betaYto = 0.005415*np.exp(V_modified)/(1 + 0.051335*np.exp(V_modified))

    Ito = Gto * Xto * Yto * (V - EK)
    df[0] += Ito
    df[11] = alphaXto*(1 - Xto) - betaXto*Xto
    df[12] = alphaYto*(1 - Yto) - betaYto*Yto

    return df

# VERIFIED
njit()
def IKp(t,var,df,EK):
    V = var[0]

    # Constants:
    KKp = 1/(1 + np.exp((7.488-V)/5.98))

    IKp = GKp * KKp * (V - EK)
    df[0] += IKp

    return df

# VERIFIED
njit()
def INaK(t,var,df):
    V = var[0]

    # Constants:
    sigma = 1/7*(np.exp(Nao/67.3) - 1)
    fNaK = 1/(1 + 0.1245*np.exp(-0.1*V*F/(R*T)) + 0.0365*sigma*np.exp(-V*F/(R*T)))
    
    INaK = INaK_max * fNaK * (1/(1 + (KmNai/Nai)**1.5)) * (Ko/(Ko + KmKo))
    df[0] += INaK

    return df

# VERIFIED
njit()
def INaCa(t,var,df):
    V = var[0]
    Cai = var[1]

    # Constants:
    VFRT = V*F/(R*T)
    frac_1 = kNaCa/(KmNa**3 + Nao**3)
    frac_2 = 1/(KmCa + Cao)
    frac_3 = 1/(1 + ksat*np.exp((eta-1)*VFRT))
    last_term = (np.exp(eta*VFRT)*(Nai**3)*Cao) - (np.exp((eta-1)*VFRT)*(Nao**3)*Cai)

    INaCa = frac_1 * frac_2 * frac_3 * last_term
    df[0] += INaCa

    return df, INaCa, VFRT

# VERIFIED
njit()
def INab(t,var,df,ENa):
    V = var[0]

    INab = GNab * (V - ENa)
    df[0] += INab

    return df

# VERIFIED
njit()
def ICab(t,var,df):
    V = var[0]
    Cai = var[1]

    # Constants:
    ECa = R*T/(2*F) * np.log(Cao/Cai)

    ICab = GCab * (V - ECa)
    df[0] += ICab

    return df, ICab

# VERIFIED
njit()
def IpCa(t,var,df):
    Cai = var[1]

    IpCa = IpCa_max * Cai / (KmpCa + Cai)
    df[0] += IpCa
    
    return df, IpCa

# VERIFIED
njit()
def ICa(t,var,df,VFRT):
    V = var[0]
    Cai = var[1]
    f = var[3]
    d = var[4]
    fCa = var[8]

    # Constants:
    VFRT2 = 2*VFRT
    first_ICa_max = PCa/Csc * 2*VFRT2*F
    second_ICa_max = (Cai*np.exp(VFRT2) - 0.341*Cao) / (np.exp(VFRT2) - 1)
    ICa_max = first_ICa_max * second_ICa_max
    finf = 1/(1+np.exp((V+12.5)/5))
    tauf = 30 + 200/(1 + np.exp((V+20)/9.5))
    dinf = 1/(1+np.exp((V+10)/-6.24))
    taud = (0.25*np.exp(-0.01*V)) / (1 + np.exp(-0.07*V)) + (0.07*np.exp(-0.05*(V+40))) / (1 + np.exp(0.05*(V+40)))
    taud = 1/taud
    fCainf = 1 / (1 + (Cai/KmfCa)**3)
    taufCa = 30

    ICa = ICa_max * f * d *fCa
    df[0] += ICa
    df[3] = (finf - f) / tauf
    df[4] = (dinf - d) / taud
    df[8] = (fCainf - fCa) / taufCa

    return df, ICa, ICa_max

# VERIFIED
njit()
def ICaK(t,var,df,VFRT,ICa_max):
    V = var[0]
    f = var[3]
    d = var[4]
    fCa = var[8]

    # Constants:
    first_ICaK = PCaK/Csc * (f*d*fCa)/(1 + ICa_max/ICahalf)
    second_ICaK = 1000*VFRT*F * ((Ki*np.exp(VFRT) - Ko) / (np.exp(VFRT) - 1))

    ICaK = first_ICaK * second_ICaK
    df[0] += ICaK
    return df

# VERIFIED
njit()
def calcium_handling(t,var,df, ICa, ICab, IpCa, INaCa):
    V = var[0]
    Cai = var[1]
    CaSR = var[2]
    f = var[3]
    d = var[4]
    fCa = var[8]

    # Constants:
    betai = 1 / (1 + (CMDNtot * KmCMDN / (KmCMDN + Cai)**2))
    gamma = 1 / (1 + (2000/CaSR)**3)
    Jrel = Prel * f*d*fCa * (gamma*CaSR - Cai)/(1 + 1.65*np.exp(V/20))
    Jup = Vup / (1 + (Kmup/Cai)**2)
    Jleak = Pleak * (CaSR - Cai)
    betaSR = 1 / (1 + (CSQNtot * KmCSQN / (KmCSQN + CaSR)**2))

    df[1] = betai * (Jrel + Jleak - Jup - (Acap*Csc/(2*F*Vmyo) * (ICa + ICab + IpCa - 2*INaCa)))
    df[2] = betaSR * (Jup - Jleak - Jrel)*Vmyo/VSR

    return df
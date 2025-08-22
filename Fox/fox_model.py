import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters
GNa = 12.8
GK1 = 2.8
GKr = 0.0136
GKs = 0.0245
GKp = 0.002216
Gto = 0.23815
GNab = 0.0031
GCab = 0.0003842

PCa = 0.0000226
PCaK = 5.79 * (10**-7)
Prel = 6
Pleak = 0.000001
INaK_max = 0.693
ICahalf = -0.265
IpCa_max = 0.05

R = 8.314
T = 310
F = 96.5
Acap = 1.534 * (10**-4)
Csc = 1
eta = 0.35

ksat = 0.2
kNaCa = 1500
KmfCa = 0.18
KmK1 = 13
KmNa = 87.5
KmCa = 1380
KmNai = 10
KmKo = 1.5
KmpCa = 0.05
Kmup = 0.32
CMDNtot = 10
CSQNtot = 10000

KmCMDN = 2
KmCSQN = 600
Vup = 0.1
Vmyo = 25.84 * (10**-6)
VSR = 2 * (10**-6)

Nai = 10
Ki = 149.4
Nao = 138
Ko = 4
Cao = 2000

def set_initial_conditions():
    """
    Key:
    0: V
    1. Ca2+i
    2. Ca2+SR
    3. f
    4. d
    5. m
    6. h
    7. j
    8. fCa
    9. XKr
    10. XKs
    11. Xto
    12. Yto
    """

    var = np.zeros(13,dtype=np.float64)
    var[0] = -94.7 
    var[1] = 0.0472 
    var[2] = 320 
    var[3] = 0.983
    var[4] = 0.0001
    var[5] = 2.4676 * (10**-4)
    var[6] = 0.99869
    var[7] = 0.99887
    var[8] = 0.942
    var[9] = 0.229
    var[10] = 0.0001
    var[11] = 3.742 * (10**-5)
    var[12] = 1
    
    return var

def function(t,var):
    df = np.zeros(13,dtype=np.float64)
    
    # df = Istim(t,var,df)
    
    df, ENa = INa(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df, EK = IK1(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df = IKr(t,var,df,EK)

    df = IKs(t,var,df)

    df = Ito(t,var,df,EK)

    df = IKp(t,var,df,EK)

    df = INaK(t,var,df)

    df, INaCa_val, VFRT = INaCa(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df = INab(t, var, df, ENa)

    df, ICab_val = ICab(t, var, df)
    # df = np.zeros(13,dtype=np.float64)

    df, IpCa_val = IpCa(t, var, df)    
    # df = np.zeros(13,dtype=np.float64)

    df, ICa_val, ICa_max = ICa(t, var, df, VFRT)
    # df = np.zeros(13,dtype=np.float64)

    df = ICaK(t, var, df, VFRT, ICa_max)

    df = calcium_handling(t, var, df, ICa_val, ICab_val, IpCa_val, INaCa_val)

    # Flip sum of voltage sum -------------
    df[0] = -df[0]
    return df

def Istim(t,var,df):
    #NOTE: RECONFIGURE
    if t == 0:
        df[0] += -80
    return df

# VERIFIED
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
def IK1(t,var,df):
    V = var[0]

    # Constants:
    EK = R*T/F*np.log(Ko/Ki)
    K1inf = 1/(2+np.exp(1.62*F/(R*T)*(V-EK)))
    
    IK1 = GK1 * K1inf * Ko / (Ko + KmK1) * (V - EK)
    df[0] += IK1

    return df, EK

# VERIFIED
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
def IKp(t,var,df,EK):
    V = var[0]

    # Constants:
    KKp = 1/(1 + np.exp((7.488-V)/5.98))

    IKp = GKp * KKp * (V - EK)
    df[0] += IKp

    return df

# VERIFIED
def INaK(t,var,df):
    V = var[0]

    # Constants:
    sigma = 1/7*(np.exp(Nao/67.3) - 1)
    fNaK = 1/(1 + 0.1245*np.exp(-0.1*V*F/(R*T)) + 0.0365*sigma*np.exp(-V*F/(R*T)))
    
    INaK = INaK_max * fNaK * (1/(1 + (KmNai/Nai)**1.5)) * (Ko/(Ko + KmKo))
    df[0] += INaK

    return df

# VERIFIED
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
def INab(t,var,df,ENa):
    V = var[0]

    INab = GNab * (V - ENa)
    df[0] += INab

    return df

# VERIFIED
def ICab(t,var,df):
    V = var[0]
    Cai = var[1]

    # Constants:
    ECa = R*T/(2*F) * np.log(Cao/Cai)

    ICab = GCab * (V - ECa)
    df[0] += ICab

    return df, ICab

# VERIFIED
def IpCa(t,var,df):
    Cai = var[1]

    IpCa = IpCa_max * Cai / (KmpCa + Cai)
    df[0] += IpCa
    
    return df, IpCa

# VERIFIED
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

# Variables
var = set_initial_conditions()
print("Initial conditions:")
print(var)

df_test = function(0, var)
print("\nDerivatives at time = 0, with initial conditions and given parameters:")
print(df_test)

t = np.linspace(0,200,20001)
values = np.zeros((13,20001))
values[:,0] = var

for i in range(0,50):
    print(f"Iteration {i}:")
    values[:,i+1] = values[:,i] + 0.01 * function(t[i], values[:,i])

print()
print(values[0,:])
print(values[0,4])


plt.plot(t[:50], values[0,:50])
plt.show()
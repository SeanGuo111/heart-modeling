module ionic_currents

export Istim, INa, IK1, IKr, IKs, Ito, IKp, INaK, INaCa, INab, ICab, IpCa, ICa, ICaK, calcium_handling_fox, calcium_handling_omichi
export N, midp, dx, D

# 2D PARAMETERS
N = 5
midp = (N+1)/2
dx = 0.5
D = 0.01

# IONIC PARAMETERS
GNa = 12.8
GK1 = 2.8
GKr = 0.0136
GKs = 0.0245
GKp = 0.002216
Gto = 0.23815
GNab = 0.0031
GCab = 0.0003842

PCa = 0.0000226
PCaK = 5.79 * (10^-7)
Prel = 6
Pleak = 0.000001
INaK_max = 0.693
ICahalf = -0.265
IpCa_max = 0.05

R = 8.314
T = 310
F = 96.5
Acap = 1.534 * (10^-4)
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
#CSQNtot = 10000 (Default)
CSQNtot = 10000

KmCMDN = 2
KmCSQN = 600
Vup = 0.1
Vmyo = 25.84 * (10^-6)
VSR = 2 * (10^-6)

Nai = 10
Ki = 149.4
Nao = 138
Ko = 4
Cao = 2000

# Omichi
k2 = 4
koff = 0.0105
v1 = 0.02
kup = 0.25
beta = 0.02
v2 = 1.5 * (10^-4)
vup = 0.1

function Istim(t,var,df,i,j)
    #NOTE: RECONFIGURE

    if (i == midp && j == midp)
        if t > 100 && t < 101
            df[1] += -80
        elseif t > 500 && t < 501
            df[1] += -80
        elseif t > 900 && t < 901
            df[1] += -80
        elseif t > 1300 && t < 1301
            df[1] += -80
        elseif t > 1700 && t < 1701
            df[1] += -80
        end
    end
    return df
end

# VERIFIED
function INa(t,var,df)
    V = var[1]
    m = var[6]
    h = var[7]    
    j = var[8]

    # Constants:
    ENa = R*T/F*log(Nao/Nai)
    alpham = 0.32 * (V + 47.13) / (1 - exp(-0.1 * (V + 47.13)))
    betam = 0.08 * exp(-V/11)
    alphah = 0.135 * exp((V + 80)/-6.8)
    betah = 7.5/(1 + exp(-0.1*(V + 11)))
    alphaj = (0.175 * exp((V + 100)/-23)) / (1 + exp(0.15 * (V + 79)))
    betaj = 0.3/(1 + exp(-0.1*(V + 32)))

    # Remember, voltage is added upon
    INa = GNa * (m^3) * h * j * (V - ENa)
    df[1] += INa
    df[6] = (alpham*(1-m)) - betam*m
    df[7] = (alphah*(1-h)) - betah*h
    df[8] = (alphaj*(1-j)) - betaj*j

    return df, ENa
end

# VERIFIED
 function IK1(t,var,df)
    V = var[1]

    # Constants:
    EK = R*T/F*log(Ko/Ki)
    K1inf = 1/(2+exp(1.62*F/(R*T)*(V-EK)))
    
    IK1 = GK1 * K1inf * Ko / (Ko + KmK1) * (V - EK)
    df[1] += IK1

    return df, EK
end

# VERIFIED
 function IKr(t,var,df,EK)
    V = var[1]
    XKr = var[10]

    # Constants:
    RV = 1/(1 + 2.5*exp(0.1*(V + 28)))
    tauKr = 43 + (1 / (exp(-5.495 + 0.1691*V) + exp(-7.677 - 0.0128*V)))
    XKrinf = 1 / (1 + exp(-2.182 - 0.1819*V))

    IKr = GKr * RV * XKr * sqrt(Ko/4) * (V - EK)
    df[1] += IKr
    df[10] = (XKrinf - XKr) / tauKr

    return df
 end

 # VERIFIED
function IKs(t,var,df)
    V = var[1]
    XKs = var[11]

    # Constants:
    EKs = R*T/F*log((Ko + 0.01833*Nao)/(Ki + 0.01833*Nai))
    XKsinf = 1 / (1 + exp((V-16)/-13.6))
    Vminus10 = V-10
    tauKs = (0.0000719*Vminus10/(1 - exp(-0.148*Vminus10))) 
    tauKs += (0.000131*Vminus10/(exp(0.0687*Vminus10) - 1))
    tauKs = 1/tauKs

    IKs = GKs * XKs^2 * (V - EKs)
    df[1] += IKs
    df[11] = (XKsinf - XKs)/tauKs

    return df
end

# VERIFIED
function Ito(t,var,df,EK)
    V = var[1]
    Xto = var[12]
    Yto = var[13]

    # Constants:
    V_modified = (V+33.5)/5
    alphaXto = 0.04516*exp(0.03577*V)
    betaXto = 0.0989*exp(-0.06237*V)
    alphaYto = 0.005415*exp(-V_modified)/(1 + 0.051335*exp(-V_modified))
    betaYto = 0.005415*exp(V_modified)/(1 + 0.051335*exp(V_modified))

    Ito = Gto * Xto * Yto * (V - EK)
    df[1] += Ito
    df[12] = alphaXto*(1 - Xto) - betaXto*Xto
    df[13] = alphaYto*(1 - Yto) - betaYto*Yto

    return df
end

# VERIFIED
function IKp(t,var,df,EK)
    V = var[1]

    # Constants:
    KKp = 1/(1 + exp((7.488-V)/5.98))

    IKp = GKp * KKp * (V - EK)
    df[1] += IKp

    return df
end

# VERIFIED
function INaK(t,var,df)
    V = var[1]

    # Constants:
    sigma = 1/7*(exp(Nao/67.3) - 1)
    fNaK = 1/(1 + 0.1245*exp(-0.1*V*F/(R*T)) + 0.0365*sigma*exp(-V*F/(R*T)))
    
    INaK = INaK_max * fNaK * (1/(1 + (KmNai/Nai)^1.5)) * (Ko/(Ko + KmKo))
    df[1] += INaK

    return df
end

# VERIFIED
function INaCa(t,var,df)
    V = var[1]
    Cai = var[2]

    # Constants:
    VFRT = V*F/(R*T)
    frac_1 = kNaCa/(KmNa^3 + Nao^3)
    frac_2 = 1/(KmCa + Cao)
    frac_3 = 1/(1 + ksat*exp((eta-1)*VFRT))
    last_term = (exp(eta*VFRT)*(Nai^3)*Cao) - (exp((eta-1)*VFRT)*(Nao^3)*Cai)

    INaCa = frac_1 * frac_2 * frac_3 * last_term
    df[1] += INaCa

    return df, INaCa, VFRT
end

# VERIFIED
function INab(t,var,df,ENa)
    V = var[1]

    INab = GNab * (V - ENa)
    df[1] += INab

    return df
end

# VERIFIED
function ICab(t,var,df)
    V = var[1]
    Cai = var[2]

    # Constants:
    ECa = R*T/(2*F) * log(Cao/Cai)

    ICab = GCab * (V - ECa)
    df[1] += ICab

    return df, ICab
end

# VERIFIED
function IpCa(t,var,df)
    Cai = var[2]

    IpCa = IpCa_max * Cai / (KmpCa + Cai)
    df[1] += IpCa
    
    return df, IpCa
end

# VERIFIED
function ICa(t,var,df,VFRT)
    V = var[1]
    Cai = var[2]
    f = var[4]
    d = var[5]
    fCa = var[9]

    # Constants:
    VFRT2 = 2*VFRT
    first_ICa_max = PCa/Csc * 2*VFRT2*F
    second_ICa_max = (Cai*exp(VFRT2) - 0.341*Cao) / (exp(VFRT2) - 1)
    ICa_max = first_ICa_max * second_ICa_max
    finf = 1/(1+exp((V+12.5)/5))
    tauf = 30 + 200/(1 + exp((V+20)/9.5))
    dinf = 1/(1+exp((V+10)/-6.24))
    taud = (0.25*exp(-0.01*V)) / (1 + exp(-0.07*V)) + (0.07*exp(-0.05*(V+40))) / (1 + exp(0.05*(V+40)))
    taud = 1/taud
    fCainf = 1 / (1 + (Cai/KmfCa)^3)
    taufCa = 30

    ICa = ICa_max * f * d *fCa
    df[1] += ICa
    df[4] = (finf - f) / tauf
    df[5] = (dinf - d) / taud
    df[9] = (fCainf - fCa) / taufCa

    return df, ICa, ICa_max
end

# VERIFIED
function ICaK(t,var,df,VFRT,ICa_max)
    V = var[1]
    f = var[4]
    d = var[5]
    fCa = var[9]

    # Constants:
    first_ICaK = PCaK/Csc * (f*d*fCa)/(1 + ICa_max/ICahalf)
    second_ICaK = 1000*VFRT*F * ((Ki*exp(VFRT) - Ko) / (exp(VFRT) - 1))

    ICaK = first_ICaK * second_ICaK
    df[1] += ICaK
    return df
end

# VERIFIED
function calcium_handling_fox(t,var,df, ICa, ICab, IpCa, INaCa)

    V = var[1]
    Cai = var[2]
    CaSR = var[3]
    f = var[4]
    d = var[5]
    fCa = var[9]

    # Constants:
    betai = 1 / (1 + (CMDNtot * KmCMDN / (KmCMDN + Cai)^2))
    gamma = 1 / (1 + (2000/CaSR)^3)
    Jrel = Prel * f*d*fCa * (gamma*CaSR - Cai)/(1 + 1.65*exp(V/20))
    Jup = Vup / (1 + (Kmup/Cai)^2)
    Jleak = Pleak * (CaSR - Cai)
    betaSR = 1 / (1 + (CSQNtot * KmCSQN / (KmCSQN + CaSR)^2))

    df[2] = betai * (Jrel + Jleak - Jup - (Acap*Csc/(2*F*Vmyo) * (ICa + ICab + IpCa - 2*INaCa)))
    df[3] = betaSR * (Jup - Jleak - Jrel)*Vmyo/VSR

    return df
end

function calcium_handling_omichi(t,var,df, ICa, ICab, IpCa, INaCa)
    Cai = var[2]
    CaSR = var[3]
    w = var[14]
    
    # Constants:
    Cs = (CaSR - Cai) / beta
    Irel = v1*w*(Cai^2)/(Cai^2 + kup^2) * (Cs - Cai)
    Ileak = v2 * (Cs - Cai)
    Iup = vup*(Cai^2) / (Cai^2 + kup^2)
    winf = 1 / (1 + (k2^2)*(Cai^2))
    tauw = winf * Cai * k2 / koff

    df[2] = Irel + Ileak - Iup - (Acap*Cs/(2*F*Vmyo) * (ICa + ICab + IpCa - 2*INaCa))
    df[3] = -(Acap*Cs/(2*F*Vmyo) * (ICa + ICab + IpCa - 2*INaCa))
    df[14] = (winf - w) / tauw

    return df
end

end
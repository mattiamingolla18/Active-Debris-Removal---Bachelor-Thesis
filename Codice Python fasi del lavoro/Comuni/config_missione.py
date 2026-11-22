import os
import numpy as np

## Parametri della missione, in un solo punto. Si cambiano senza toccare il file:
##   ADR_ATMOSFERA=alta python 5_rientro/rientro.py   (bassa, moderata, alta; piano_bassa, piano_media, piano_alta)
##   ADR_J2=1 python 5_rientro/rientro.py             (J2 attivo)
## Un valore scritto male ferma subito lo script con un messaggio, invece di cambiare caso in silenzio.
ATMOSFERE_VALIDE=("bassa","moderata","alta","piano_bassa","piano_media","piano_alta")

C_D=2.2
A_TARGET=4.0 # m^2
ISP=300.0 # s
RHO_CHASER=150.0 # kg/m^3, densita' media del cubo equivalente al chaser
J2=1.08263e-3

valore_j2=os.environ.get("ADR_J2","0")
if valore_j2 not in ("0","1"):
    raise ValueError(f"ADR_J2='{valore_j2}' non valido: usa 0 (spento) oppure 1 (attivo)")
J2_ATTIVO=valore_j2=="1"

## Area media di un cubo in rotazione libera: superficie/4 = 1,5 L^2 (teorema di Cauchy)
def area_chaser(m_secco,rho=RHO_CHASER):
    L=(m_secco/rho)**(1/3)
    return 1.5*L**2

## Massa al lancio a ritroso dalla massa a secco, Delta-V in m/s.
## La riserva resta a bordo e va spinta anch'essa: ogni kg di riserva costa (K-1) kg di propellente.
def massa_al_lancio(m_secco,m_target,dv_phasing,dv_deorbit,isp,g0,margine=0.0):
    c=isp*g0
    k_phasing=np.exp(dv_phasing/c)
    k_deorbit=np.exp(dv_deorbit/c)
    K=k_phasing*k_deorbit
    consumo=((m_secco+m_target)*K-m_target*k_phasing-m_secco)/(1-margine*(K-1))
    return m_secco+(1+margine)*consumo

## Tsiolkovsky in avanti: i tre consumi e la massa finale del chaser
def bilancio_in_avanti(m_lancio,m_target,dv1,dv2,dv3,isp,g0):
    c=isp*g0
    m1=m_lancio*np.exp(-dv1/c)
    m2=m1*np.exp(-dv2/c)
    prima_dv3=m2+m_target
    dopo_dv3=prima_dv3*np.exp(-dv3/c)
    return m_lancio-m1,m1-m2,prima_dv3-dopo_dv3,dopo_dv3-m_target

## Target: Starlink v1.5, massa a secco (catalogo McDowell). Chaser dimensionato a ritroso.
M_TARGET=275.0 # kg
M_SECCO_CHASER=150.0 # kg
A_CHASER=area_chaser(M_SECCO_CHASER)
MARGINE=0.10 # riserva: 10% del propellente consumato

## Atmosfera: NRLMSISE-00, attivita' solare moderata (F10.7 = 140, Ap = 15)
ATMOSFERA=os.environ.get("ADR_ATMOSFERA","moderata")
if ATMOSFERA not in ATMOSFERE_VALIDE:
    raise ValueError(f"ADR_ATMOSFERA='{ATMOSFERA}' non valido: usa uno tra {ATMOSFERE_VALIDE}")

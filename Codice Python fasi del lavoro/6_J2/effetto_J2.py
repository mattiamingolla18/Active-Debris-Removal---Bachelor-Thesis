import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","Comuni"))
import percorsi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import config_missione as cfg
import atmosfera_nrlmsise as ATM
import rientro as P4
import rientro_msis as PM

## Effetto di J2 nel modello 2D equatoriale. Nel piano J2 e' una forza centrale,
## a_r=-(mu/r^2)*(1+1.5*J2*(R/r)^2): energia e momento angolare si conservano, quindi apsidi,
## tempi e archi si ricavano esattamente. La propagazione numerica e' il controllo indipendente.
J2=cfg.J2
mu=P4.mu
R_e=P4.R_earth
r_c=P4.r_chaser
r_t=P4.r_target
r_d=P4.r_deorbit
fase_0=np.radians(P4.delta_phi_0)
tolleranze=dict(method='DOP853',rtol=1e-12,atol=1e-10)

def gradi_giorno(w):
    return np.degrees(w)*86400

## Velocita' angolare [rad/s] dell'orbita circolare equatoriale di raggio r
def omega_circolare(r,j2=J2):
    return np.sqrt(mu/r**3*(1+1.5*j2*(R_e/r)**2))

## Velocita' tangenziale [km/s] in r1 perche' l'altro apside sia in r2
def v_apside(r1,r2,j2=J2):
    k=0.5*mu*j2*R_e**2
    return np.sqrt(2*(mu*(1/r1-1/r2)+k*(1/r1**3-1/r2**3))/(1-(r1/r2)**2))

## Apsidi di uno stato tangenziale: radici di 2E r^3 + 2 mu r^2 - h^2 r + 2k = 0.
## La radice piu' piccola non e' fisica, le altre due sono perigeo e apogeo.
def apsidi(r,v,j2=J2):
    k=0.5*mu*j2*R_e**2
    h=r*v
    E=0.5*v*v-mu/r-k/r**3
    return np.sort(np.roots([2*E,2*mu,-h*h,2*k]).real)

## Tempo [s] e angolo [rad] da un apside all'altro, per quadratura di Gauss-Legendre.
## Con r=r_m-D*cos(psi) la singolarita' agli estremi sparisce.
def arco(r1,r2,v1,j2=J2):
    k=0.5*mu*j2*R_e**2
    h=r1*v1
    E=0.5*v1*v1-mu/r1-k/r1**3
    r3=apsidi(r1,v1,j2)[0]
    x,w=np.polynomial.legendre.leggauss(400)
    psi=0.5*np.pi*(x+1)
    w=0.5*np.pi*w
    lo,hi=min(r1,r2),max(r1,r2)
    r=0.5*(lo+hi)-0.5*(hi-lo)*np.cos(psi)
    f=np.sqrt(r**3/(-2*E*(r-r3)))
    return float(np.sum(w*f)),float(np.sum(w*f*h/r**2))

## Gravita' con J2, senza drag
def gravita(j2):
    def f(t,s):
        x,y,vx,vy=s
        r=np.hypot(x,y)
        c=-mu/r**3*(1+1.5*j2*(R_e/r)**2)
        return [vx,vy,c*x,c*y]
    return f

def propaga(stato,durata,j2):
    return solve_ivp(gravita(j2),[0,durata],stato,**tolleranze).y[:,-1]

def spinta_tangenziale(stato,dv):
    s=np.array(stato,float)
    s[2:]*=1+dv/np.hypot(s[2],s[3])
    return s

### 1. Precessione dell'apside sull'ellisse 550 x 200 km, senza drag ###
## Si misura rispetto a una direzione inerziale: d(omega+Omega)/dt = 1,5 n J2 (R/p)^2.
## La formula 3 n J2 (R/p)^2 del piano di chiusura e' il solo d(omega)/dt: vale il doppio.
v_a=v_apside(r_t,r_d)
a=0.5*(r_t+r_d)
e=(r_t-r_d)/(r_t+r_d)
p=a*(1-e**2)
n=np.sqrt(mu/a**3)

def pericentro(t,s):
    return s[0]*s[2]+s[1]*s[3]
pericentro.direction=1

sol=solve_ivp(gravita(J2),[0,30*86400],[r_t,0.0,0.0,v_a],events=pericentro,**tolleranze)
t_p=sol.t_events[0]
s_p=sol.y_events[0]
angolo_p=np.unwrap(np.arctan2(s_p[:,1],s_p[:,0]))
w_num=float(np.polyfit(t_p,angolo_p,1)[0])
semi_periodo,semi_angolo=arco(r_t,r_d,v_a)
w_quad=(semi_angolo-np.pi)/semi_periodo
w_formula=float(1.5*n*J2*(R_e/p)**2)
w_piano=float(3*n*J2*(R_e/p)**2)

righe=[]
for nome,w in [("Propagazione DOP853",w_num),("Quadratura esatta",w_quad),
               ("1,5 n J2 (R/p)^2 = d(omega + Omega)/dt",w_formula),
               ("3 n J2 (R/p)^2 = d(omega)/dt (piano di chiusura)",w_piano)]:
    print(f"{nome}: {gradi_giorno(w):.4f} gradi/giorno")
    righe.append([nome,float(f"{w:.6e}"),round(float(gradi_giorno(w)),4)])
percorsi.salva_csv("J2_precessione.csv",["metodo","rateo_rad_s","rateo_gradi_giorno"],righe)

### 2. Phasing 400 -> 550 km: errore al rendez-vous di piani di volo diversi ###
## Con J2 un'orbita circolare ha velocita' r*Omega, non sqrt(mu/r). Correggere solo il wait time
## non basta, perche' l'arco di trasferimento non e' 180 gradi: il piano completo ricalcola tutto.
o_c=omega_circolare(r_c)
o_t=omega_circolare(r_t)
n_c=omega_circolare(r_c,0.0)
n_t=omega_circolare(r_t,0.0)

## Piano kepleriano: wait, tempo di volo, arco, dv1, dv2
v1=v_apside(r_c,r_t,0.0)
tof=np.pi*np.sqrt((0.5*(r_c+r_t))**3/mu)
piano_kep=((fase_0-(np.pi-n_t*tof))/(n_c-n_t),tof,np.pi,v1-r_c*n_c,r_t*n_t-v1*r_c/r_t)

## Solo il wait time corretto con Omega, come proposto nel piano di chiusura
piano_wait=((fase_0-(np.pi-o_t*tof))/(o_c-o_t),)+piano_kep[1:]

## Piano completo con J2
v1=v_apside(r_c,r_t)
tof,angolo=arco(r_c,r_t,v1)
piano_j2=((fase_0-(angolo-o_t*tof))/(o_c-o_t),tof,angolo,v1-r_c*o_c,r_t*o_t-v1*r_c/r_t)

## Vola un piano: errore radiale, lungo la traccia, distanza [km] e velocita' relativa [m/s]
def errore_rendezvous(piano,j2,v_c,v_t):
    wait,tof,arco_t,dv1,dv2=piano
    c=spinta_tangenziale(propaga([r_c,0.0,0.0,v_c],wait,j2),dv1)
    c=propaga(c,tof,j2)
    t=propaga([r_t*np.cos(fase_0),r_t*np.sin(fase_0),-v_t*np.sin(fase_0),v_t*np.cos(fase_0)],wait+tof,j2)
    u_r=t[:2]/np.hypot(t[0],t[1])
    d=c[:2]-t[:2]
    dv=spinta_tangenziale(c,dv2)[2:]-t[2:]
    return float(d@u_r),float(d@np.array([-u_r[1],u_r[0]])),float(np.hypot(*d)),float(np.hypot(*dv))*1e3

v_kep=(r_c*n_c,r_t*n_t)
v_j2=(r_c*o_c,r_t*o_t)
casi=[("A. Kepleriano, modello senza J2 (controllo)",piano_kep,0.0,v_kep),
      ("B. Kepleriano, modello con J2",piano_kep,J2,v_j2),
      ("C. Solo wait time con J2 (piano di chiusura)",piano_wait,J2,v_j2),
      ("D. Piano completo con J2",piano_j2,J2,v_j2),
      ("E. Kepleriano, J2 acceso sulle velocita' kepleriane",piano_kep,J2,v_kep)]

righe=[]
for nome,piano,j2,(v_c,v_t) in casi:
    errore=errore_rendezvous(piano,j2,v_c,v_t)
    print(f"{nome}: distanza {errore[2]:.3f} km")
    righe.append([nome,round(float(piano[0]),2),round(float(piano[1]),2),round(float(piano[3])*1e3,3),
                  round(float(piano[4])*1e3,3),round(float(np.degrees(piano[2])),3)]+[round(m,3) for m in errore])
percorsi.salva_csv("J2_rendezvous.csv",
                   ["piano","wait_s","TOF_s","dV1_m_s","dV2_m_s","arco_gradi","errore_radiale_km",
                    "errore_lungo_traccia_km","distanza_km","dV_relativa_m_s"],righe)

### 3. Rientro con e senza J2 ###
## Lo stato iniziale deve essere coerente con J2: lasciando quello kepleriano il perigeo
## scende a circa 181 km e il rientro si accorcia per un artefatto, non per un effetto fisico.
v_a_kep=v_apside(r_t,r_d,0.0)
dv3_kep=r_t*n_t-v_a_kep
stati={"senza J2":(0.0,v_a_kep),
       "J2 su stato kepleriano (artefatto)":(J2,v_a_kep),
       "J2, target circolare, stesso dV3":(J2,r_t*o_t-dv3_kep),
       "J2, perigeo a 200 km":(J2,v_a)}
for nome,(j2,v) in stati.items():
    print(f"{nome}: perigeo {apsidi(r_t,v,j2)[1]-R_e:.2f} km")

## Tempo di rientro con la densita' attiva in rientro.py
def giorni_rientro(B_c,j2,v):
    P4.j2=j2
    return P4.tempo_rientro(B_c,[r_t,0.0,0.0,v])

## Tempo di rientro con il modello completo: con J2 il perigeo attraversa i tempi locali
def giorni_completo(scenario,B_c,j2):
    f107,ap=ATM.SCENARI[scenario]
    stato=np.array(PM.state0,float)
    if j2:
        stato[2:]*=v_a/np.hypot(stato[2],stato[3])
    sol=solve_ivp(PM.make_rhs(f107,ap,B_c,j2),[0,400*86400],stato,method='DOP853',rtol=1e-8,atol=1e-9,events=PM.reentry)
    return sol.t_events[0][0]/86400

m_base=cfg.M_SECCO_CHASER
B_c_base=float(P4.B_c_sistema(m_base))

righe=[]

## Caso base nei tre scenari
risultati={}
for scenario in ATM.SCENARI:
    P4.h_atm,P4.rho_atm,P4.H_atm=ATM.carica_tabella(scenario)
    risultati[scenario]={nome:giorni_rientro(B_c_base,j2,v) for nome,(j2,v) in stati.items()}
    for nome,t in risultati[scenario].items():
        righe.append(["tabella",f"chaser {m_base:.0f} kg",scenario,nome,round(B_c_base,2),round(t,3),
                      round(100*(t/risultati[scenario]["senza J2"]-1),2)])

## Altri chaser
for m in [100.0,200.0]:
    B_c=float(P4.B_c_sistema(m))
    for scenario in ATM.SCENARI:
        P4.h_atm,P4.rho_atm,P4.H_atm=ATM.carica_tabella(scenario)
        t0=giorni_rientro(B_c,*stati["senza J2"])
        t1=giorni_rientro(B_c,*stati["J2, perigeo a 200 km"])
        righe.append(["tabella",f"chaser {m:.0f} kg",scenario,"senza J2",round(B_c,2),round(t0,3),0.0])
        righe.append(["tabella",f"chaser {m:.0f} kg",scenario,"J2, perigeo a 200 km",round(B_c,2),round(t1,3),round(100*(t1/t0-1),2)])

## Modello con il tempo locale reale
for scenario in ATM.SCENARI:
    t0=giorni_completo(scenario,B_c_base,0.0)
    t1=giorni_completo(scenario,B_c_base,J2)
    print(f"Tempo locale, {scenario}: senza J2 {t0:.2f}, con J2 {t1:.2f} giorni")
    righe.append(["tempo locale",f"chaser {m_base:.0f} kg",scenario,"senza J2",round(B_c_base,2),round(t0,3),0.0])
    righe.append(["tempo locale",f"chaser {m_base:.0f} kg",scenario,"J2, perigeo a 200 km",round(B_c_base,2),round(t1,3),round(100*(t1/t0-1),2)])

P4.j2=0.0
percorsi.salva_csv("J2_rientro.csv",
                   ["atmosfera","caso","scenario","stato_iniziale","B_C_kg_m2","rientro_giorni","differenza_percentuale"],righe)

## Figura 23: precessione dell'apside
plt.figure(figsize=(7.4,4.7))
giorni=np.array([0.0,30.0])
plt.plot(giorni,gradi_giorno(w_piano)*giorni,'--',color='tab:orange',label=f'Formula del piano: {gradi_giorno(w_piano):.1f} gradi/giorno')
plt.plot(giorni,gradi_giorno(w_formula)*giorni,color='tab:blue',label=f'Formula corretta: {gradi_giorno(w_formula):.1f} gradi/giorno')
plt.plot(t_p[::30]/86400,np.degrees(angolo_p[::30]-angolo_p[0]),'o',color='navy',label='Propagazione numerica')
plt.xlabel('Tempo [giorni]')
plt.ylabel('Rotazione della linea degli apsidi [gradi]')
plt.grid(True,linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(percorsi.DIR_GRAFICI_ANALISI,'23_J2_precessione_apside.png'),dpi=200)

## Figura 24: rientro con e senza J2 nei tre scenari
plt.figure(figsize=(7.4,4.7))
scenari=list(ATM.SCENARI)
serie=[("senza J2","Senza J2"),("J2, perigeo a 200 km","Con J2, perigeo a 200 km"),
       ("J2 su stato kepleriano (artefatto)","J2 senza correggere lo stato iniziale (artefatto)")]
for i,(chiave,nome) in enumerate(serie):
    plt.bar(np.arange(3)+(i-1)*0.26,[risultati[s][chiave] for s in scenari],0.24,label=nome)
plt.xticks(np.arange(3),[f"Attivita' {s}" for s in scenari])
plt.ylabel('Tempo di rientro [giorni]')
plt.grid(True,axis='y',linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(percorsi.DIR_GRAFICI_ANALISI,'24_J2_rientro.png'),dpi=200)

import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","Comuni"))
import percorsi
from phasing import *
from deorbiting import *
from atmosfera_nrlmsise import carica_tabella
import config_missione as cfg
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Modello semplificato: gravita' kepleriana (Terra puntiforme) + drag atmosferico, nel piano
## equatoriale. J2 e' opzionale (ADR_J2=1, vedi config_missione.py) e di default e' spento.
## mu e raggio terrestre sono quelli delle fasi 2-3, cosi' tutto il progetto usa le stesse costanti.
R_earth=r_earth # km
j2=cfg.J2 if cfg.J2_ATTIVO else 0.0

## Orbita di rientro ereditata dalla fase 3 (apogeo = quota del target, perigeo = quota di deorbiting):
apogee=r_target-r_earth   # 550 km
perigee=r_deorbit-r_earth # 200 km

r_apog=apogee+R_earth
r_perig=perigee+R_earth
a_deorbit=(r_apog+r_perig)/2
v_apog=np.sqrt(mu*(2/r_apog-1/a_deorbit))
if j2:
    ## Con J2 la velocita' kepleriana porterebbe il perigeo a circa 181 km e accorcerebbe il rientro
    ## per un artefatto. Questa e' la velocita' che lo tiene a 200 km: nel piano J2 e' una forza
    ## centrale, quindi basta imporre energia e momento angolare tra apogeo e perigeo.
    k_j2=0.5*mu*j2*R_earth**2
    v_apog=np.sqrt(2*(mu*(1/r_apog-1/r_perig)+k_j2*(1/r_apog**3-1/r_perig**3))/(1-(r_apog/r_perig)**2))
T_orbit=(2*np.pi*np.sqrt(a_deorbit**3)/(np.sqrt(mu)))

state_vector=[r_apog,0,0,v_apog] ## [x,y,vx,vy]: apogeo dell'ellisse (550 km), velocita' perpendicolare al raggio

w_earth=7.2921e-5 # rad/s, velocita' angolare di rotazione terrestre (l'atmosfera ruota con la Terra)

### Modello atmosferico: NRLMSISE-00 in tabelle ###
## Esponenziale a tratti: rho(h)=rho_i*exp(-(h-h_i)/H_i). Quote h_i, densita' rho_i e altezze di scala H_i
## vengono da tabelle_NRLMSISE00.csv, per lo scenario di attivita' solare scelto in config_missione.py.
h_atm,rho_atm,H_atm=carica_tabella(cfg.ATMOSFERA)

def density(h):
    """Densita' atmosferica [kg/m^3] alla quota geometrica h [km].

    Il modello e' definito tra la prima e l'ultima quota della tabella: fuori da questo intervallo la quota
    viene saturata agli estremi, cosi' la funzione resta definita anche durante
    il rientro (h<0) senza interrompere l'integratore.
    """
    z=np.clip(h,h_atm[0],h_atm[-1])
    i=np.searchsorted(h_atm,z,side='right')-1  ## intervallo [h_i, h_i+1) che contiene z
    i=min(i,len(H_atm)-1)                      ## all'ultima quota si usa l'ultimo intervallo
    rho=rho_atm[i]*np.exp(-(z-h_atm[i])/H_atm[i])
    return rho

## b_c e' un argomento: solve_ivp lo passa con args=(b_c,), senza modificare variabili globali
def kepler_2d(t,state_vector,b_c):
    x,y,vx,vy=state_vector[0],state_vector[1],state_vector[2],state_vector[3]
    h=np.sqrt(x**2+y**2)-R_earth
    rho=density(h)
    r=np.sqrt(x**2+y**2)
    ax=-mu*x/(r**3)
    ay=-mu*y/(r**3)
    a_j2=-1.5*j2*mu*R_earth**2/r**5 ## J2 nel piano: a = -(3/2) J2 mu R^2 / r^4, verso il centro

    v_atm=[-w_earth*y,w_earth*x]
    v_rel=(np.subtract([vx,vy],v_atm))*1000
    v_rel_norm=(np.sqrt(v_rel[0]**2+v_rel[1]**2))
    v_rel_hat=v_rel/v_rel_norm

    ax_drag=(-0.5*rho*(v_rel_norm**2*v_rel_hat[0])*(1/b_c))/1000
    ay_drag=(-0.5*rho*(v_rel_norm**2*v_rel_hat[1])*(1/b_c))/1000
    ax_total=ax+ax_drag+a_j2*x
    ay_total=ay+ay_drag+a_j2*y
    return [vx,vy,ax_total,ay_total]

## Evento di rientro: interrompe l'integrazione quando la quota scende sotto h_reentry
h_reentry=100 # km, quota convenzionale di rientro (linea di Karman)
## solve_ivp passa args=(b_c,) anche agli eventi: b_c va accettato anche se qui non serve
def reentry(t,state_vector,b_c):
    x,y=state_vector[0],state_vector[1]
    return np.sqrt(x**2+y**2)-R_earth-h_reentry
reentry.terminal=True
reentry.direction=-1

t_max=3*365*24*3600 # limite superiore della propagazione: 3 anni

## Tempo di rientro [giorni] per un dato coefficiente balistico, dall'ellisse di deorbiting
def tempo_rientro(b_c,stato=state_vector):
    sol=solve_ivp(kepler_2d,[0,t_max],stato,method='DOP853',rtol=1e-8,atol=1e-9,events=reentry,args=(b_c,))
    if sol.t_events[0].size==0:
        raise RuntimeError(f"Nessun rientro entro {t_max/86400:.0f} giorni con B_C = {b_c:.2f} kg/m^2")
    return sol.t_events[0][0]/86400

## Tempo di rientro naturale [anni] da orbita circolare, mediato sull'orbita (il tempo scala con B_C)
def tempo_naturale(b_c,h0=550.0):
    anno=365.25*86400
    def dadt(t,a):
        a_m=a[0]*1000
        v=np.sqrt(mu*1e9/a_m)
        f=(1-w_earth*a[0]/(v/1000))**2
        return [-density(a[0]-R_earth)*np.sqrt(mu*1e9*a_m)*f/b_c/1000]
    def fine(t,a):
        return a[0]-R_earth-h_reentry
    fine.terminal=True
    sol=solve_ivp(dadt,[0,200*anno],[R_earth+h0],events=fine,rtol=1e-9,atol=1e-6,max_step=86400*5)
    if sol.t_events[0].size==0:
        raise RuntimeError(f"Nessun rientro naturale entro 200 anni con B_C = {b_c:.2f} kg/m^2")
    return sol.t_events[0][0]/anno

## Integrator verification: quota dopo una singola orbita
check=solve_ivp(kepler_2d,[0,T_orbit],state_vector,method='DOP853',rtol=1e-10,atol=1e-12,args=(B_c,))
r_check=np.sqrt(check.y[0,-1]**2+check.y[1,-1]**2)
print(f"Altitude after one orbit (with drag): {r_check-R_earth:.4f} km")

## Propagazione fino al rientro
t_eval=np.arange(0,t_max,60) # campionamento ogni 60 s per i grafici
solution=solve_ivp(kepler_2d,[0,t_max],state_vector,method='DOP853',rtol=1e-8,atol=1e-9,
                   t_eval=t_eval,events=reentry,args=(B_c,))

x=solution.y[0]
y=solution.y[1]
r=np.sqrt(x**2+y**2)
h=r-R_earth
t_days=solution.t/86400

## Rivoluzioni effettive: l'angolo polare "srotolato" cresce di 2*pi a ogni giro.
## (t_decay/T_orbit le sottostimerebbe: il periodo si accorcia durante il decadimento)
n_rev=np.unwrap(np.arctan2(y,x))[-1]/(2*np.pi)

if solution.t_events[0].size>0:
    t_decay=solution.t_events[0][0]
    print(f"Decay time from {apogee:.0f}x{perigee:.0f} km to {h_reentry} km: "
          f"{t_decay/86400:.2f} days ({t_decay/3600:.1f} hours)")
    print(f"Number of revolutions before reentry: {n_rev:.0f}")
else:
    print(f"No reentry within {t_max/86400:.0f} days: final altitude {h[-1]:.2f} km")

print(f"Ballistic coefficient used: {B_c:.2f} kg/m^2")

## Grafico dell'andamento della quota (solo se il file viene lanciato da solo)
if __name__=='__main__':
    plt.figure(figsize=(10,5))
    plt.plot(t_days,h,linewidth=0.5)
    plt.axhline(h_reentry,color='r',linestyle='--',linewidth=1,label=f'Reentry ({h_reentry} km)')
    plt.xlabel('Time [days]')
    plt.ylabel('Altitude [km]')
    plt.title(f'Orbital decay - initial orbit {apogee:.0f}x{perigee:.0f} km, B$_C$={B_c:.1f} kg/m$^2$')
    plt.grid(True,linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(percorsi.DIR_GRAFICI_ANALISI,'decay_altitude.png'),dpi=150)
    plt.show()

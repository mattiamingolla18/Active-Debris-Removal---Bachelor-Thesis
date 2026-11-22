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

## Rientro nei tre scenari di attivita' solare, con due modelli di densita':
## tabella  = rho(h) mediata sui tempi locali. E' l'unica adatta al rientro naturale, che dura
##            anni e su orbita circolare attraversa tutti i tempi locali a ogni giro.
## completo = NRLMSISE-00 a ogni passo. Il perigeo resta a un tempo locale fissato dalla
##            geometria: e' il confronto diretto con FreeFlyer.
anno=365.25*86400
C_D=cfg.C_D
m_T=cfg.M_TARGET
A_T=cfg.A_TARGET
masse_secco=[50.0,100.0,150.0,200.0,250.0]
aree_target=[2.0,4.0,6.0]

## Fa usare a rientro.py la tabella di densita' di uno scenario
def usa_scenario(scenario):
    P4.h_atm,P4.rho_atm,P4.H_atm=ATM.carica_tabella(scenario)

## Propagazione con il modello completo
def propaga_completo(scenario,B_c,t_eval=None):
    f107,ap=ATM.SCENARI[scenario]
    return solve_ivp(PM.make_rhs(f107,ap,B_c),[0,400*86400],PM.state0,method='DOP853',rtol=1e-8,atol=1e-9,
                     t_eval=t_eval,events=PM.reentry)

## Riga con il rientro nei due modelli
def riga(gruppo,caso,scenario,B_c):
    tabella=P4.tempo_rientro(B_c)
    completo=propaga_completo(scenario,B_c).t_events[0][0]/86400
    print(f"{gruppo} {caso} {scenario}: B_C {B_c:.2f}, tabella {tabella:.2f}, completo {completo:.2f} giorni")
    return [gruppo,caso,scenario,round(B_c,3),round(tabella,3),round(completo,3),"giorni"]

righe=[]
soluzioni={}
for scenario in ATM.SCENARI:
    usa_scenario(scenario)
    for m_secco in masse_secco:
        righe.append(riga("sistema",f"chaser {m_secco:.0f} kg secco",scenario,P4.B_c_sistema(m_secco)))

    for A in aree_target:
        if A!=A_T:
            righe.append(riga("sistema_area",f"A target {A:.0f} m2",scenario,P4.B_c_sistema(cfg.M_SECCO_CHASER,A)))
        B_c=m_T/(C_D*A)
        righe.append(riga("solo_target",f"A target {A:.0f} m2",scenario,B_c))
        t_nat=P4.tempo_naturale(B_c)
        print(f"naturale A = {A:.0f} m^2 {scenario}: {t_nat:.2f} anni")
        righe.append(["naturale",f"A target {A:.0f} m2",scenario,round(B_c,3),round(t_nat,3),"","anni"])

    ## Traiettoria del sistema per la figura 20, campionata ogni 600 s
    soluzioni[scenario]=propaga_completo(scenario,P4.B_c,np.arange(0,400*86400,600.0))

## Scenari del piano di chiusura (solo tabella), per confronto con la scelta ECSS
B_c_target=m_T/(C_D*A_T)
for scenario in ATM.SCENARI_PIANO:
    usa_scenario(scenario)
    t_sistema=P4.tempo_rientro(P4.B_c)
    t_target=P4.tempo_rientro(B_c_target)
    t_nat=P4.tempo_naturale(B_c_target)
    print(f"{scenario}: sistema {t_sistema:.2f}, solo target {t_target:.2f} giorni, naturale {t_nat:.2f} anni")
    righe.append(["piano","sistema",scenario,round(P4.B_c,3),round(t_sistema,3),"","giorni"])
    righe.append(["piano","solo target",scenario,round(B_c_target,3),round(t_target,3),"","giorni"])
    righe.append(["piano","naturale",scenario,round(B_c_target,3),round(t_nat,3),"","anni"])

## Vela di drag sul target: il tempo di rientro e' proporzionale a B_C, cioe' a 1/A
for scenario in ATM.SCENARI:
    usa_scenario(scenario)
    anni=P4.tempo_naturale(B_c_target)
    t_adr=soluzioni[scenario].t_events[0][0]/anno
    for nome,obiettivo in [("25 anni",25.0),("5 anni",5.0),("come ADR",t_adr)]:
        vela=max(A_T*anni/obiettivo-A_T,0.0)
        print(f"{scenario}, rientro {nome}: vela {vela:.1f} m^2")
        righe.append(["vela",nome,scenario,"",round(vela,3),"","m2"])

## Verifica del modello mediato: rientro naturale numerico con attivita' alta
usa_scenario("alta")
r0=P4.R_earth+550.0
numerico=P4.tempo_rientro(B_c_target,[r0,0.0,0.0,np.sqrt(P4.mu/r0)])
mediato=P4.tempo_naturale(B_c_target)*anno/86400
print(f"Rientro naturale, attivita' alta: numerico {numerico:.1f}, mediato {mediato:.1f} giorni")
righe.append(["verifica","naturale numerico","alta",round(B_c_target,3),round(numerico,3),"","giorni"])

percorsi.salva_csv("rientro_scenari_solari.csv",
                   ["gruppo","caso","scenario","B_C_kg_m2","valore_tabella","valore_completo","unita"],righe)

## Figura 19: densita' NRLMSISE-00 nei tre scenari
quote=np.linspace(100,600,501)
plt.figure(figsize=(10,5.4))
for scenario,(f107,ap) in ATM.SCENARI.items():
    usa_scenario(scenario)
    plt.semilogy(quote,[P4.density(q) for q in quote],label=f"Attivita' {scenario}: F10.7 = {f107:.0f}, Ap = {ap:.0f}")
plt.xlabel('Quota [km]')
plt.ylabel('Densita\' [kg/m$^3$]')
plt.grid(True,linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(percorsi.DIR_GRAFICI_ANALISI,'19_densita_NRLMSISE_scenari.png'),dpi=200)

## Figura 20: apogeo (continuo) e perigeo (tratteggiato) del sistema nei tre scenari
plt.figure(figsize=(10,5.4))
for scenario,sol in soluzioni.items():
    x,y,vx,vy=sol.y
    r=np.hypot(x,y)
    a=1/(2/r-(vx**2+vy**2)/P4.mu)
    e=np.sqrt(np.maximum(1-(x*vy-y*vx)**2/(P4.mu*a),0))
    sopra=a*(1-e)-P4.R_earth>=100 # sotto i 100 km il perigeo osculante non ha piu' senso
    giorni=sol.t[sopra]/86400
    linea,=plt.plot(giorni,(a*(1+e)-P4.R_earth)[sopra],label=f"Attivita' {scenario}: {sol.t_events[0][0]/86400:.1f} giorni")
    plt.plot(giorni,(a*(1-e)-P4.R_earth)[sopra],'--',color=linea.get_color())
plt.axhline(100,color='r',linestyle='--',linewidth=1,label='Rientro (100 km)')
plt.xlabel('Tempo dal burn di deorbiting [giorni]')
plt.ylabel('Quota [km]')
plt.grid(True,linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(percorsi.DIR_GRAFICI_ANALISI,'20_rientro_scenari_solari.png'),dpi=200)

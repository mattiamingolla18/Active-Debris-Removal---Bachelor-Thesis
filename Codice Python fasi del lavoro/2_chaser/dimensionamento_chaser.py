import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","Comuni"))
import percorsi
import numpy as np
import matplotlib.pyplot as plt
import config_missione as cfg
import rientro as P4

## Dimensionamento del chaser: massa al lancio a ritroso dalla massa a secco.
## I Delta-V dipendono solo dalle orbite: cambiano solo propellente, masse e B_C.
dv1=abs(P4.delta_v1_chaser)*1000 # m/s
dv2=abs(P4.delta_v2_chaser)*1000
dv3=abs(P4.delta_v3)*1000
g=P4.g
m_T=cfg.M_TARGET
A_T=cfg.A_TARGET
C_D=cfg.C_D
margine=cfg.MARGINE
m_base=cfg.M_SECCO_CHASER
masse_secco=[50.0,100.0,150.0,200.0,250.0]
isp_lista=[220.0,300.0] # s: monopropellente, bipropellente

## Il consumo e' lineare nella massa a secco: (K-1)*m_secco + quota fissa dovuta al target
for isp in isp_lista:
    k12=np.exp((dv1+dv2)/(isp*g))
    k3=np.exp(dv3/(isp*g))
    print(f"Isp {isp:.0f} s: {k12*k3-1:.4f} kg per kg a secco + {m_T*k12*(k3-1):.2f} kg fissi per il target")

## Riga della tabella: dal bilancio delle masse a B_C e tempo di rientro
def riga(tipo,m_secco,isp,m_lancio,bilancio,a_c,margine):
    p1,p2,p3,m_fine=bilancio
    consumo=p1+p2+p3
    riserva=margine*consumo
    B_c=(m_fine+m_T)/(C_D*(A_T+a_c))
    rientro=P4.tempo_rientro(B_c)
    print(f"{tipo}: secco {m_secco:.1f} kg, Isp {isp:.0f} s, lancio {m_lancio:.1f} kg, "
          f"propellente {consumo+riserva:.2f} kg, B_C {B_c:.2f}, rientro {rientro:.2f} giorni")
    valori=[m_secco,isp,m_lancio,p1,p2,p3,consumo,riserva,consumo+riserva,(consumo+riserva)/m_lancio,
            m_lancio/m_T,m_lancio-p1-p2+m_T,m_fine+m_T,a_c,B_c,rientro]
    return [tipo]+[round(float(v),3) for v in valori]

## Sweep a ritroso su massa a secco e Isp
righe=[]
for isp in isp_lista:
    for m_secco in masse_secco:
        m_lancio=cfg.massa_al_lancio(m_secco,m_T,dv1+dv2,dv3,isp,g,margine)
        bilancio=cfg.bilancio_in_avanti(m_lancio,m_T,dv1,dv2,dv3,isp,g)
        righe.append(riga("a ritroso",m_secco,isp,m_lancio,bilancio,cfg.area_chaser(m_secco),margine))

percorsi.salva_csv("chaser_sweep.csv",
                   ["tipo","m_secco_kg","Isp_s","m_lancio_kg","prop_dV1_kg","prop_dV2_kg","prop_dV3_kg","consumo_kg",
                    "riserva_kg","propellente_caricato_kg","frazione_propellente","rapporto_lancio_target",
                    "m_sistema_cattura_kg","m_sistema_finale_kg","A_chaser_m2","B_C_kg_m2","rientro_giorni"],righe)

## Sensibilita' del caso base: nome, massa del target, area del target, densita' del chaser, margine
varianti=[("Caso base",m_T,A_T,cfg.RHO_CHASER,margine),
          ("A target = 2 m2",m_T,2.0,cfg.RHO_CHASER,margine),
          ("A target = 6 m2",m_T,6.0,cfg.RHO_CHASER,margine),
          ("Densita' chaser 200 kg/m3",m_T,A_T,200.0,margine),
          ("Densita' chaser 400 kg/m3",m_T,A_T,400.0,margine),
          ("Target 290 kg",290.0,A_T,cfg.RHO_CHASER,margine),
          ("Senza riserva",m_T,A_T,cfg.RHO_CHASER,0.0)]

righe=[]
for nome,m_t,a_t,rho,marg in varianti:
    m_lancio=cfg.massa_al_lancio(m_base,m_t,dv1+dv2,dv3,cfg.ISP,g,marg)
    p1,p2,p3,m_fine=cfg.bilancio_in_avanti(m_lancio,m_t,dv1,dv2,dv3,cfg.ISP,g)
    a_c=cfg.area_chaser(m_base,rho)
    B_c=(m_fine+m_t)/(C_D*(a_t+a_c))
    rientro=P4.tempo_rientro(B_c)
    print(f"{nome}: lancio {m_lancio:.2f} kg, B_C {B_c:.2f}, rientro {rientro:.2f} giorni")
    righe.append([nome,round(float(m_lancio),2),round(float((1+marg)*(p1+p2+p3)),2),round(float(a_c),2),
                  round(float(B_c),2),round(float(rientro),2)])

percorsi.salva_csv("chaser_sensibilita_caso_base.csv",
                   ["variante","m_lancio_kg","propellente_caricato_kg","A_chaser_m2","B_C_kg_m2","rientro_giorni"],righe)

## Tabella della slide 8, caso base
m_lancio=cfg.massa_al_lancio(m_base,m_T,dv1+dv2,dv3,cfg.ISP,g,margine)
p1,p2,p3,m_fine=cfg.bilancio_in_avanti(m_lancio,m_T,dv1,dv2,dv3,cfg.ISP,g)
m1=m_lancio-p1
m2=m1-p2
righe=[["dV1 phasing (400 km)",round(dv1,2),round(float(p1),2),round(float(m1),2)],
       ["dV2 rendez-vous (550 km)",round(dv2,2),round(float(p2),2),round(float(m2),2)],
       ["dV3 deorbiting (550 km)",round(dv3,2),round(float(p3),2),round(float(m_fine+m_T),2)],
       ["Totale missione",round(dv1+dv2+dv3,2),round(float(p1+p2+p3),2),""]]
percorsi.salva_csv("chaser_tabella_slide8.csv",["manovra","dV_m_s","propellente_kg","massa_dopo_kg"],righe)
print(f"Deorbiting: {100*p3/(p1+p2+p3):.1f} % del consumo, la massa da spingere diventa {(m2+m_T)/m2:.2f} volte quella del chaser")

## Figura 21: propellente caricato contro massa a secco, disegnata nella misura della slide 8
## (5,5 x 3,49 pollici), cosi' in presentazione i caratteri restano leggibili
with plt.rc_context({'font.size':11}):
    plt.figure(figsize=(5.5,3.49))
    x=np.linspace(0,260,200)
    for isp,colore in [(220.0,'tab:blue'),(300.0,'navy')]:
        plt.plot(x,[cfg.massa_al_lancio(m,m_T,dv1+dv2,dv3,isp,g,margine)-m for m in x],color=colore,
                 label=f'Isp = {isp:.0f} s, con riserva del 10 %')
        plt.plot(masse_secco,[cfg.massa_al_lancio(m,m_T,dv1+dv2,dv3,isp,g,margine)-m for m in masse_secco],'o',color=colore)
    plt.plot(m_base,m_lancio-m_base,'*',color='r',markersize=14,label='caso base')
    plt.annotate(f'{m_lancio-m_base:.1f} kg su {m_lancio:.0f} kg al lancio'.replace('.',','),
                 xy=(m_base,m_lancio-m_base),xytext=(150,5),ha='center',
                 arrowprops=dict(arrowstyle='-',color='0.4',lw=0.8))
    plt.xlabel('Massa a secco del chaser [kg]')
    plt.ylabel('Propellente caricato [kg]')
    plt.xlim(0,260)
    plt.ylim(bottom=0)
    plt.grid(True,linestyle=':')
    plt.legend(loc='upper left',fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(percorsi.DIR_GRAFICI_ANALISI,'21_propellente_vs_massa_secco.png'),dpi=200)

## Figura 22: tempo di rientro contro massa a secco, per tre aree del target
plt.figure(figsize=(7.4,4.7))
masse=masse_secco
for A,colore in [(2.0,'navy'),(4.0,'tab:blue'),(6.0,'lightskyblue')]:
    plt.plot(masse,[P4.tempo_rientro(P4.B_c_sistema(m,A)) for m in masse],'-o',color=colore,label=f'A target = {A:.0f} m$^2$')
plt.plot(m_base,P4.tempo_rientro(P4.B_c),'*',color='r',markersize=16,label='caso base')
plt.xlabel('Massa a secco del chaser [kg]')
plt.ylabel('Tempo di rientro [giorni]')
plt.ylim(bottom=0)
plt.grid(True,linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(percorsi.DIR_GRAFICI_ANALISI,'22_rientro_vs_massa_secco.png'),dpi=200)

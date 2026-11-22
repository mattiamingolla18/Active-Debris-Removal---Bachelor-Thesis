import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","Comuni"))
import percorsi
import config_missione as cfg
import rientro as P4

## Controlli da rilanciare dopo ogni modifica al codice
dv1=abs(P4.delta_v1_chaser)*1000 # m/s
dv2=abs(P4.delta_v2_chaser)*1000
dv3=abs(P4.delta_v3)*1000

## 1. Caso di riferimento: chaser di 150 kg a secco, NRLMSISE-00 moderata in tabelle, senza J2
print(f"Delta-V totale: {dv1+dv2+dv3:.3f} m/s (atteso 182.392)")
print(f"Massa al lancio: {P4.mass_chaser:.3f} kg (attesa 171.324)")
print(f"Propellente consumato: {P4.total_propellant_burned:.3f} kg (atteso 19.386)")
print(f"B_C del sistema: {P4.B_c:.3f} kg/m^2 (atteso 35.284)")
print(f"Rientro: {P4.t_decay/86400:.3f} giorni (atteso 15.525)")
assert abs(dv1+dv2+dv3-182.392)<0.001
assert abs(P4.mass_chaser-171.324)<0.001
assert abs(P4.total_propellant_burned-19.386)<0.001
assert abs(P4.B_c-35.284)<0.001
assert abs(P4.t_decay/86400-15.525)<0.001

## 2. Controllo indipendente della massa al lancio: secco 100 kg, target 275 kg, Isp = 300 s,
## g0 = 9,80665, senza riserva. Valori attesi calcolati a mano con
## m_lancio = [(m_secco + m_T) e^(dV3/c) - m_T] e^((dV1+dV2)/c) e Tsiolkovsky per ogni burn.
g0=9.80665
m_lancio=cfg.massa_al_lancio(100.0,275.0,dv1+dv2,dv3,300.0,g0)
p1,p2,p3,m_fine=cfg.bilancio_in_avanti(m_lancio,275.0,dv1,dv2,dv3,300.0,g0)
print(f"Massa al lancio: {m_lancio:.3f} kg (attesa 116.071), propellente {p1:.3f} + {p2:.3f} + {p3:.3f} kg "
      f"(attesi 1.639, 1.607, 12.824)")
assert abs(m_lancio-116.071)<0.001
assert abs(p1-1.639)<0.001 and abs(p2-1.607)<0.001 and abs(p3-12.824)<0.001
assert abs(m_fine-100.0)<1e-6

## 3. A ritroso e in avanti devono dare la stessa massa finale, con riserva del 10%
for m_secco in [50.0,150.0,250.0]:
    for isp in [220.0,300.0]:
        m_lancio=cfg.massa_al_lancio(m_secco,275.0,dv1+dv2,dv3,isp,P4.g,0.10)
        p1,p2,p3,m_fine=cfg.bilancio_in_avanti(m_lancio,275.0,dv1,dv2,dv3,isp,P4.g)
        assert abs(m_fine-(m_secco+0.10*(p1+p2+p3)))<1e-6

## 4. Le cartelle dei moduli devono essere le stesse per Python (percorsi.py) e per l'editor
## (pyrightconfig.json): se divergono, VS Code torna a segnalare import "non risolti"
import json
import re
with open(os.path.join(percorsi.RADICE,"pyrightconfig.json"),encoding="utf-8") as f:
    testo=re.sub(r"^\s*//.*$","",f.read(),flags=re.MULTILINE) # toglie i commenti, che JSON non ammette
assert json.loads(testo)["extraPaths"]==percorsi.CARTELLE_MODULI, "pyrightconfig.json e percorsi.py non coincidono"

print("Tutti i controlli superati.")

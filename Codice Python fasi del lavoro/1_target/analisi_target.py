import os
import sys
## Rende visibile la cartella Comuni per poter importare percorsi.py, che poi aggiunge le altre
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","Comuni"))
import percorsi
import config_missione as cfg
import rientro as P4

## Analisi del target: coefficiente balistico, rientro naturale e sensibilita'. I tempi usano l'atmosfera attiva (predefinita: NRLMSISE-00 moderata)
print(f"\n--- Analisi del target: atmosfera {cfg.ATMOSFERA} ---")
C_D=cfg.C_D
m_T=cfg.M_TARGET
anno=365.25*86400
dv1=abs(P4.delta_v1_chaser)*1000 # m/s
dv2=abs(P4.delta_v2_chaser)*1000
dv3=abs(P4.delta_v3)*1000

## Coefficiente balistico del target
print(f"B_C target: {m_T/(C_D*cfg.A_TARGET):.2f} kg/m^2")

## Rientro della missione: serve per calcolare l'area equivalente all'ADR
t_adr=P4.tempo_rientro(P4.B_c)
print(f"Rientro del sistema (B_C = {P4.B_c:.2f}): {t_adr:.2f} giorni")

## Sensibilita' su area e massa. Rotazione libera: il pannello e' una lastra 9,0 x 2,8 m
## (catalogo McDowell) e l'area media di una lastra e' meta' della faccia.
casi=[("A = 2 m2",m_T,2.0),
      ("A = 4 m2 (riferimento)",m_T,4.0),
      ("A = 6 m2",m_T,6.0),
      ("Massa 290 kg (v1.5 recenti)",290.0,4.0),
      ("Rotazione libera, pannello integro",m_T,9.0*2.8/2)]

righe=[]
for nome,m,A in casi:
    B_c=m/(C_D*A)
    t_nat=P4.tempo_naturale(B_c)
    t_ellisse=P4.tempo_rientro(B_c)
    ## Il tempo e' proporzionale a B_C: da qui l'area totale che porta il rientro al valore voluto
    k=t_nat/B_c
    A_5=k*m/(C_D*5)
    vela_5=max(A_5-A,0.0)
    A_adr=k*m/(C_D*(t_adr*86400/anno))
    print(f"{nome}: B_C {B_c:.2f}, naturale {t_nat:.2f} anni, solo target {t_ellisse:.2f} giorni, "
          f"vela per 5 anni {vela_5:.2f} m^2, area pari all'ADR {A_adr:.0f} m^2")
    righe.append([nome,float(m),float(A),round(B_c,2),round(t_nat,2),round(t_ellisse,2),round(A_5,2),
                  round(vela_5,2),round(A_adr,0),"si" if t_nat<=25 else "no"])

percorsi.salva_csv("target_sensibilita.csv",
                   ["caso","massa_kg","area_m2","B_C_kg_m2","rientro_naturale_anni","solo_target_ellisse_giorni",
                    "area_totale_5_anni_m2","vela_5_anni_m2","area_pari_ADR_m2","entro_25_anni"],righe)

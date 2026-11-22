import os
import numpy as np

## Tabelle di densita' NRLMSISE-00 per gli scenari di attivita' solare.
## Lanciato da solo chiama pymsis e rigenera il CSV. Gli altri script leggono il CSV con
## carica_tabella(): l'integratore non chiama mai pymsis e resta veloce.
## Tabella: quote 80-1000 km, latitudine 0, media su 24 longitudini (cioe' sui tempi locali),
## data 12 gennaio 2020 (centro del rientro in FreeFlyer), F10.7 = F10.7 medio, Ap costante.
FILE_TABELLE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tabelle_NRLMSISE00.csv")

## Scenari (F10.7, Ap): ECSS-E-ST-10-04C, gli stessi dei file di flusso solare di FreeFlyer
SCENARI={"bassa":(65.0,0.0),"moderata":(140.0,15.0),"alta":(250.0,45.0)}
## Scenari proposti nel piano di chiusura: restano come controllo della scelta fatta
SCENARI_PIANO={"piano_bassa":(70.0,5.0),"piano_media":(150.0,15.0),"piano_alta":(230.0,30.0)}

quote=np.arange(80.0,1001.0,10.0) # km
longitudini=np.arange(0.0,360.0,15.0) # deg, a UT = 0 sono i 24 tempi locali
data_tabella="2020-01-12T00:00"

## pymsis si blocca sulla lettera accentata di UNIVERSITA' nel percorso dei suoi file di parametri.
## NRLMSISE-00 (version=0) non li legge, quindi basta dargli un percorso senza accenti.
def importa_pymsis():
    import pymsis
    import pymsis.msis
    pymsis.msis._MSIS_PARAMETER_PATH="./"
    return pymsis

## Altezze di scala H_i=(h_i+1-h_i)/ln(rho_i/rho_i+1), come le tabelle di rientro.py
def altezze_scala(h,rho):
    return (h[1:]-h[:-1])/np.log(rho[:-1]/rho[1:])

## Restituisce (h, rho, H) di uno scenario, nello stesso formato delle tabelle di rientro.py
def carica_tabella(scenario):
    with open(FILE_TABELLE,encoding="utf-8") as f:
        righe=[r for r in f if not r.startswith("#")]
    colonne=righe[0].strip().split(",")
    dati=np.loadtxt(righe[1:],delimiter=",")
    h=dati[:,0]
    rho=dati[:,colonne.index(scenario)]
    return h,rho,altezze_scala(h,rho)

if __name__=="__main__":
    pymsis=importa_pymsis()
    tutti={**SCENARI,**SCENARI_PIANO}
    colonne=[]
    for f107,ap in tutti.values():
        out=pymsis.calculate(np.datetime64(data_tabella),longitudini,0.0,quote,
                             f107s=[f107],f107as=[f107],aps=[[ap]*7],version=0)
        colonne.append(np.squeeze(out[...,pymsis.Variable.MASS_DENSITY]).mean(axis=0))
    intestazione=("# NRLMSISE-00 (pymsis, version=0): densita' totale [kg/m^3], latitudine 0,\n"
                  f"# media su {longitudini.size} tempi locali, data {data_tabella}, F10.7 = F10.7 media.\n"
                  "# Scenari (F10.7, Ap): "+"; ".join(f"{n} = ({f:.0f}, {a:.0f})" for n,(f,a) in tutti.items())+"\n"
                  "h_km,"+",".join(tutti))
    np.savetxt(FILE_TABELLE,np.column_stack([quote,*colonne]),delimiter=",",header=intestazione,comments="",
               fmt=["%.1f"]+["%.6e"]*len(tutti),encoding="utf-8")

    ## Controlli: la densita' scende con la quota e l'attivita' alta e' sempre la piu' densa
    bassa,moderata,alta=(carica_tabella(s)[1] for s in SCENARI)
    print("Densita' decrescente con la quota:",all(np.all(np.diff(r)<0) for r in (bassa,moderata,alta)))
    print("Quote dove non vale alta > moderata > bassa:",quote[~((alta>moderata)&(moderata>bassa))])

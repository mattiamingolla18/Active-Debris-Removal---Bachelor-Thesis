import os
import sys

## Cartelle del progetto: gli script stanno in cartelle diverse ma si importano a vicenda
RADICE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # "Codice Python fasi del lavoro"
RADICE_REPO=os.path.dirname(RADICE)                                 # cartella principale del repository
DIR_RISULTATI=os.path.join(RADICE,"Risultati")
FF_DIR=os.path.join(RADICE_REPO,"FREEFLYER SIMULATION")

## Figure: una sottocartella per uso, dentro "Grafici e immagini"
DIR_IMMAGINI=os.path.join(RADICE_REPO,"Grafici e immagini")
DIR_GRAFICI_TESI=os.path.join(DIR_IMMAGINI,"tesi")                   # 01-10, 8_grafici/grafici_tesi.py
DIR_GRAFICI_PRESENTAZIONE=os.path.join(DIR_IMMAGINI,"presentazione") # 11-18, 8_grafici/grafici_presentazione.py
DIR_GRAFICI_ANALISI=os.path.join(DIR_IMMAGINI,"analisi")             # 19-24 e decay_altitude, script delle fasi

## Cartelle che contengono moduli importati da altri script.
## Questa lista serve a Python quando ESEGUE il codice. L'editor (VS Code/Pylance) non esegue
## nulla e legge la stessa lista da pyrightconfig.json ("extraPaths"): se aggiungi una cartella,
## aggiornala in entrambi i posti (controlli/test_riferimento.py controlla che coincidano).
CARTELLE_MODULI=["Comuni","3_phasing","4_deorbiting","5_rientro","7_validazione_freeflyer"]

for cartella in CARTELLE_MODULI:
    percorso=os.path.join(RADICE,cartella)
    if percorso not in sys.path: # evita di aggiungere la stessa cartella piu' volte
        sys.path.append(percorso)

## Le cartelle di output vengono create se mancano, cosi' il salvataggio non fallisce
for cartella in [DIR_RISULTATI,DIR_GRAFICI_TESI,DIR_GRAFICI_PRESENTAZIONE,DIR_GRAFICI_ANALISI]:
    os.makedirs(cartella,exist_ok=True)

## Salva una tabella in Risultati (separatore ; e virgola decimale: si apre in Excel)
def salva_csv(nome,intestazione,righe):
    with open(os.path.join(DIR_RISULTATI,nome),"w",encoding="utf-8-sig",newline="") as f:
        f.write(";".join(intestazione)+"\r\n")
        for riga in righe:
            f.write(";".join(str(v).replace(".",",") if isinstance(v,float) else str(v) for v in riga)+"\r\n")

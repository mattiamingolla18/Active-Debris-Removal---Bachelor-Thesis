# Active Debris Removal – Tesi triennale in Ingegneria Aerospaziale

Simulazione della rimozione attiva di un satellite Starlink in LEO (Università di Padova, Dinamica del volo spaziale):
rendezvous del chaser con il target, aggancio, burn di deorbiting e rientro passivo per drag atmosferico.
I calcoli sono in Python; FreeFlyer serve per la simulazione della missione e la validazione incrociata.

## Struttura della cartella

Le fonti bibliografiche (PDF e note) sono fuori dal repository, nella cartella `TESI TRIENNALE/Fonti bibliografiche`:
non vanno pubblicate su GitHub.

Il codice è diviso in una cartella per fase della missione, in ordine cronologico. Le cartelle 3-5 formano il modello
di base (phasing → deorbiting → rientro): ogni fase importa i risultati della precedente.

```
Active-Debris-Removal-Bachelor-Thesis/
├── Codice Python fasi del lavoro/
│   ├── Comuni/                        Usato da tutti gli script
│   │   ├── config_missione.py         Tutti i parametri della missione
│   │   └── percorsi.py                Cartelle del progetto, rende visibili i moduli, salva i CSV
│   ├── 1_target/
│   │   └── analisi_target.py          B_C del target, rientro naturale, sensibilità su area e massa, vela
│   ├── 2_chaser/
│   │   └── dimensionamento_chaser.py  Massa al lancio a ritroso dalla massa a secco, sweep su massa e Isp
│   ├── 3_phasing/
│   │   ├── phasing.py                 Hohmann 400 -> 550 km, wait time, Δv1, Δv2, massa al lancio
│   │   └── phasing_params.txt         Valori inseriti in FreeFlyer (wait time, TOF, Δv1-3)
│   ├── 4_deorbiting/
│   │   └── deorbiting.py              Aggancio, Δv3 di deorbiting (perigeo 200 km), coefficiente balistico
│   ├── 5_rientro/
│   │   ├── rientro.py                 Propagazione del rientro (DOP853), tempi di rientro, J2 opzionale
│   │   ├── rientro_scenari_solari.py  Rientri nei tre scenari di attività solare
│   │   ├── rientro_msis.py            Rientro con NRLMSISE-00 a ogni passo (modello completo)
│   │   ├── atmosfera_nrlmsise.py      Genera e legge le tabelle di densità NRLMSISE-00
│   │   └── tabelle_NRLMSISE00.csv     Tabelle di densità per scenario di attività solare
│   ├── 6_J2/
│   │   └── effetto_J2.py              Effetto di J2 su apside, rendez-vous e rientro
│   ├── 7_validazione_freeflyer/
│   │   ├── freeflyer_scenari.py       Corse FreeFlyer nei tre scenari di attività solare
│   │   └── freeflyer_J2.py            Controllo di J2 con FreeFlyer
│   ├── 8_grafici/
│   │   ├── grafici_tesi.py            Figure 01-10
│   │   ├── grafici_presentazione.py   Figure 11-18
│   │   └── grafici_readme.py          Figure in inglese per il README di GitHub
│   ├── controlli/
│   │   └── test_riferimento.py        Controlli di non regressione
│   ├── Risultati/                     Tabelle CSV (target_..., chaser_..., rientro_..., J2_...)
│   ├── pyrightconfig.json             Cartelle dei moduli per l'editor (VS Code/Pylance)
│   └── requirements.txt               Librerie Python necessarie
├── Grafici e immagini/
│   ├── tesi/                          Figure 01-10 (grafici_tesi.py)
│   ├── presentazione/                 Figure 11-18 (grafici_presentazione.py)
│   ├── analisi/                       Figure 19-24 e decay_altitude.png (script delle fasi)
│   ├── freeflyer/                     Screenshot della simulazione FreeFlyer
│   └── readme/                        Figure in inglese del README (grafici_readme.py)
├── FREEFLYER SIMULATION/
│   ├── Mattia Mingolla Tesi_FreeFlyer7.10.0.MissionPlan   Mission Plan in uso
│   ├── flusso_solare_ECSS_*.txt       Attività solare per i tre scenari
│   ├── Starlink Epoch.txt             Epoca di rientro a 100 km dell'ultima corsa (generata da FreeFlyer)
│   └── Risultati/                     Corse nominali (A), scenari solari (B), controllo J2 (C)
├── README.md                          Descrizione in inglese per GitHub
├── README_IT.md                       Questo file
└── pyrightconfig.json                 Come sopra, per quando si apre in VS Code la cartella principale
```

`grafici_tesi.py` legge i file di `FREEFLYER SIMULATION` per il grafico di validazione (09): non rinominare quella cartella né i file di output.
Nei moduli importati si usano gli alias `P2`, `P3`, `P4` (phasing, deorbiting, rientro).

## Come eseguire il codice Python

Dalla cartella `Codice Python fasi del lavoro`, con l'ambiente virtuale `.venv` attivo:

```
python -m pip install -r requirements.txt
python 3_phasing/phasing.py
python 4_deorbiting/deorbiting.py
python 5_rientro/rientro.py
python 1_target/analisi_target.py                   (~1 minuto)
python 2_chaser/dimensionamento_chaser.py           (~3 minuti)
python 5_rientro/rientro_scenari_solari.py
python 6_J2/effetto_J2.py                           (~2 minuti)
python 8_grafici/grafici_tesi.py                    (~3 minuti)
python 8_grafici/grafici_presentazione.py           (~3-5 minuti)
python 8_grafici/grafici_readme.py                  (~10 secondi)
python controlli/test_riferimento.py
```

Ogni script importa i risultati delle fasi precedenti tramite `Comuni/percorsi.py`, quindi funziona da qualunque cartella.
L'editor non esegue il codice e trova le stesse cartelle tramite `pyrightconfig.json`: se si aggiunge una cartella di moduli,
va aggiornata sia in `CARTELLE_MODULI` (`percorsi.py`) sia in `extraPaths` (`test_riferimento.py` controlla che coincidano).
Un valore non valido di `ADR_ATMOSFERA` o `ADR_J2` ferma lo script con un messaggio che elenca i valori ammessi.
Il caso predefinito è il chaser di 150 kg a secco, con NRLMSISE-00 moderata e senza J2. Per cambiarlo senza modificare i file (Git Bash):

```
ADR_ATMOSFERA=alta python 2_chaser/dimensionamento_chaser.py
ADR_J2=1 python 5_rientro/rientro.py
```

Su questo PC un criterio di Windows blocca a intermittenza alcune DLL di SciPy. Se uno script si ferma con
"un criterio di controllo dell'applicazione ha bloccato il file", basta rilanciarlo.

## Simulazione FreeFlyer

Il Mission Plan è stato creato con FreeFlyer 7.10 e gira anche con FreeFlyer 8 (conversione automatica all'apertura).
I file di output sono scritti con percorsi assoluti nella cartella `FREEFLYER SIMULATION` su Google Drive
(`G:\Il mio Drive\UNIVERSITÀ\TESI TRIENNALE\...`): se la cartella viene spostata, vanno aggiornati i percorsi
delle tre istruzioni `Maneuver` e del `Report Starlink.Epoch` nel Mission Plan.

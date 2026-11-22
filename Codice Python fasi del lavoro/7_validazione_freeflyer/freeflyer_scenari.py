## Corse FreeFlyer nei tre scenari di attivita' solare, da riga di comando (ff.exe).
## Per ogni scenario crea una COPIA del Mission Plan in Risultati/B_NRLMSISE_scenari,
## la esegue e legge l'epoca di rientro. Il Mission Plan principale non viene toccato.
## Uso: python 7_validazione_freeflyer/freeflyer_scenari.py
import csv
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Comuni"))
import percorsi
import phasing as P2
import config_missione as cfg

FF_DIR = percorsi.FF_DIR
PIANO = os.path.join(FF_DIR, "Mattia Mingolla Tesi_FreeFlyer7.10.0.MissionPlan")
OUT_DIR = os.path.join(FF_DIR, "Risultati", "B_NRLMSISE_scenari")
FF_EXE = r"C:\Program Files\a.i. solutions, Inc\FreeFlyer 7.10.1.60799527 (64-Bit)\ff.exe"

SCENARI = {"bassa": "flusso_solare_ECSS_bassa_F107_65_Ap_0.txt",
           "moderata": "flusso_solare_ECSS_moderata_F107_140_Ap_15.txt",
           "alta": "flusso_solare_ECSS_alta_F107_250_Ap_45.txt"}

## Caso simulato: chaser di 150 kg a secco. Il Mission Plan principale contiene gia' le masse
## (chaser al lancio, target) e le aree di drag di questo caso: prepara_piano controlla che
## coincidano con config_missione.py, cosi' una modifica dei parametri non passa inosservata.
CASO = f"chaser_{cfg.M_SECCO_CHASER:.0f}kg"
ATTESI = {"VehicleDryMass": sorted([float(P2.mass_chaser), cfg.M_TARGET]),
          "DragArea": sorted([cfg.A_TARGET, cfg.A_CHASER])}
AREA_SISTEMA = cfg.A_TARGET + cfg.A_CHASER  # area di drag dopo la cattura

TAI_MENO_UTC = 37.0  # s: Starlink.Epoch e' riportata in TAI, le epoche dei burn in UTC


## Come str.replace, ma si ferma se il testo atteso non c'e': la copia sarebbe sbagliata
def sostituisci(testo, vecchio, nuovo, minimo=1):
    if testo.count(vecchio) < minimo:
        raise ValueError(f"Nel Mission Plan non trovo: {vecchio}")
    return testo.replace(vecchio, nuovo)


def prepara_piano(scenario):
    cartella = os.path.join(OUT_DIR, f"{CASO}_{scenario}")
    os.makedirs(cartella, exist_ok=True)
    testo = open(PIANO, encoding="utf-8", newline="").read()

    testo = sostituisci(testo, os.path.join(FF_DIR, "Risultati", "A_nominale_phasing_kepleriano") + "\\",
                        cartella + "\\")
    for nome in ("Cattura.txt", "Starlink Epoch.txt"):
        testo = sostituisci(testo, os.path.join(FF_DIR, nome), os.path.join(cartella, nome))
    testo = sostituisci(testo, SCENARI["moderata"], SCENARI[scenario])

    ## Masse e aree del Mission Plan = quelle del codice
    for tag, attesi in ATTESI.items():
        valori = sorted(float(x) for x in re.findall(rf"<{tag} [^>]*>([\d.]+)</{tag}>", testo))
        if len(valori) != 2 or any(abs(v - a) > 1e-3 for v, a in zip(valori, attesi)):
            raise ValueError(f"Nel Mission Plan {tag} = {valori}, nel codice {attesi}: aggiorna il Mission Plan")
    for veicolo in ("Chaser", "Starlink"):
        if f"{veicolo}.DragArea = {AREA_SISTEMA};" not in testo:
            raise ValueError(f"Nel Mission Plan manca {veicolo}.DragArea = {AREA_SISTEMA};")

    ## Nessun output deve restare nelle cartelle del progetto gia' esistenti
    for percorso in re.findall(r'[A-Z]:\\[^<"&\r\n]*', testo):
        if percorso.startswith(FF_DIR) and not (percorso.startswith(cartella) or "flusso_solare" in percorso):
            raise ValueError(f"Output non reindirizzato: {percorso}")

    file_piano = os.path.join(cartella, f"{caso}_{scenario}.MissionPlan")
    with open(file_piano, "w", encoding="utf-8", newline="") as fh:
        fh.write(testo)
    return cartella, file_piano


## Tempo di rientro dal burn di deorbiting [giorni], massa del sistema e propellente del burn
def leggi_risultati(cartella):
    burn = open(os.path.join(cartella, "4_Burn3_deorbiting_perigeo_200km.txt"), encoding="utf-8").read()
    mjd_burn = float(re.search(r"Modified Julian:\s*([\d.]+)", burn).group(1))
    massa = float(re.search(r"End mass:\s*([\d.]+)", burn).group(1))
    consumo = float(re.search(r"Mass used:\s*([\d.]+)", burn).group(1))
    epoca = open(os.path.join(cartella, "Starlink Epoch.txt"), encoding="utf-8").read()
    mjd_rientro = float(re.findall(r"\d{5}\.\d+", epoca)[-1])
    return mjd_rientro - TAI_MENO_UTC / 86400.0 - mjd_burn, massa, consumo


def esegui(scenario):
    cartella, file_piano = prepara_piano(scenario)
    subprocess.run([FF_EXE, "-mp", file_piano, "-hwm"], check=True, capture_output=True, timeout=600)
    return leggi_risultati(cartella)


if __name__ == "__main__":
    file_csv = os.path.join(OUT_DIR, "rientri_freeflyer.csv")
    righe = []
    for scenario in SCENARI:
        t, massa, consumo = esegui(scenario)
        righe.append((CASO, scenario, f"{t:.4f}", f"{massa:.3f}", f"{consumo:.3f}"))
        print(f"{CASO:16s} {scenario:9s} rientro {t:7.3f} giorni, massa sistema {massa:.2f} kg")

    with open(file_csv, "w", encoding="utf-8", newline="") as fh:
        scrittore = csv.writer(fh)
        scrittore.writerow(("caso", "scenario", "rientro_giorni", "massa_sistema_kg", "propellente_burn3_kg"))
        scrittore.writerows(righe)

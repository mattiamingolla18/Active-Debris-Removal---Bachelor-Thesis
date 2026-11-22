## Controllo di 6_J2/effetto_J2.py con FreeFlyer, da riga di comando (ff.exe). Tre corse sull'ellisse
## 550 x 200 km senza drag: stato coerente con J2, stato kepleriano, Terra puntiforme.
## Crea COPIE del Mission Plan in Risultati/C_J2_controllo: il Mission Plan principale non viene toccato.
## Uso: python 7_validazione_freeflyer/freeflyer_J2.py
import csv
import math
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Comuni"))
import percorsi
import rientro as P4
import config_missione as cfg
from freeflyer_scenari import FF_DIR, FF_EXE, PIANO

OUT_DIR = os.path.join(FF_DIR, "Risultati", "C_J2_controllo")
GIORNI = 15

MU, RE = P4.mu, P4.R_earth
R_A, R_P = P4.r_apog, P4.r_perig
K = 0.5 * MU * cfg.J2 * RE ** 2
V_KEP = math.sqrt(MU * (2 / R_A - 2 / (R_A + R_P)))
V_J2 = math.sqrt(2 * (MU * (1 / R_A - 1 / R_P) + K * (1 / R_A ** 3 - 1 / R_P ** 3))
                 / (1 - (R_A / R_P) ** 2))

## nome, velocita' all'apogeo [km/s], tipo di campo (0 punto materiale, 1 solo zonali), grado, ordine
CASI = [("stato_J2", V_J2, 1, 2, 0), ("stato_kepleriano", V_KEP, 1, 2, 0),
        ("puntiforme", V_KEP, 0, 0, 0)]


## La Terra e' il terzo corpo della lista FreeFlyer
def imposta_terra(blocco, tag, valore):
    m = re.search(r"(<" + tag + r"[^>]*>(?:\s*<value>[^<]*</value>){2}\s*<value>)[^<]*(</value>)", blocco)
    if m is None:
        raise ValueError(f"Nel Mission Plan non trovo: {tag}")
    return blocco[:m.start()] + m.group(1) + str(valore) + m.group(2) + blocco[m.end():]


def prepara_piano(nome, v, tipo, grado, ordine):
    testo = open(PIANO, encoding="utf-8", newline="").read()
    i = testo.index('<ForceModel name="Starlink_ForceModel"')
    j = testo.index("</ForceModel>", i)
    blocco = testo[i:j].replace("<Drag>true</Drag>", "<Drag>false</Drag>")
    for tag, valore in (("PlanetFieldType", tipo), ("PlanetFieldDegree", grado), ("PlanetFieldOrder", ordine)):
        blocco = imposta_terra(blocco, tag, valore)
    testo = testo[:i] + blocco + testo[j:]

    stato = os.path.join(OUT_DIR, f"{nome}_stato.txt")
    if os.path.exists(stato):
        os.remove(stato)
    script = f"""Starlink.Cd = 0.0;
Starlink.X = {R_A!r};
Starlink.Y = 0.0;
Starlink.Z = 0.0;
Starlink.VX = 0.0;
Starlink.VY = {v!r};
Starlink.VZ = 0.0;
While (Starlink.ElapsedTime < TIMESPAN({GIORNI} days));
Step Starlink;
Report Starlink.Epoch<24,13>, Starlink.X<22,9>, Starlink.Y<22,9>, Starlink.VX<22,12>, Starlink.VY<22,12> to "{stato}";
End;"""
    comando = ('<FreeForm enabled="true"><Label>FreeForm</Label><Bookmarks /><Breakpoints />'
               f"<FreeFormScript><![CDATA[{script}]]></FreeFormScript></FreeForm>")
    testo = re.sub(r"<ProjectCommands>.*</ProjectCommands>",
                   lambda m: f"<ProjectCommands>{comando}</ProjectCommands>", testo, flags=re.S)
    testo = re.sub(r"<ProjectScript><!\[CDATA\[.*?\]\]></ProjectScript>",
                   lambda m: f"<ProjectScript><![CDATA[{script}]]></ProjectScript>", testo, flags=re.S)
    piano = os.path.join(OUT_DIR, f"{nome}.MissionPlan")
    with open(piano, "w", encoding="utf-8", newline="") as fh:
        fh.write(testo)
    return piano, stato


## Precessione [gradi/giorno] e quota media del perigeo [km] dagli stati a ogni passo
def analizza(file_stato):
    righe = []
    with open(file_stato, encoding="utf-8", errors="ignore") as fh:
        for riga in fh:
            campi = riga.split()
            if len(campi) == 5:
                try:
                    righe.append([float(c) for c in campi])
                except ValueError:
                    pass                # intestazione del report
    d = np.array(righe)
    t = (d[:, 0] - d[0, 0]) * 86400.0
    x, y, vx, vy = d[:, 1:].T
    r_dot_v, r = x * vx + y * vy, np.hypot(x, y)
    i = np.where((r_dot_v[:-1] < 0) & (r_dot_v[1:] >= 0))[0]      # passaggi al perigeo
    f = -r_dot_v[i] / (r_dot_v[i + 1] - r_dot_v[i])
    angolo = np.unwrap(np.arctan2(y[i] + f * (y[i + 1] - y[i]), x[i] + f * (x[i + 1] - x[i])))
    rateo = np.polyfit(t[i] + f * (t[i + 1] - t[i]), angolo, 1)[0]
    minimi = []
    for k in i:
        c = np.polyfit(t[k - 1:k + 3] - t[k], r[k - 1:k + 3], 2)
        minimi.append(c[2] - c[1] ** 2 / (4 * c[0]))
    return math.degrees(rateo) * 86400.0, float(np.mean(minimi)) - RE, len(i)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    righe = []
    for nome, v, tipo, grado, ordine in CASI:
        piano, stato = prepara_piano(nome, v, tipo, grado, ordine)
        subprocess.run([FF_EXE, "-mp", piano, "-hwm"], check=True, capture_output=True, timeout=600)
        rateo, perigeo, passaggi = analizza(stato)
        os.remove(stato)
        print(f"{nome:18s} precessione {rateo:8.4f} gradi/giorno, perigeo {perigeo:8.3f} km ({passaggi} perigei)")
        righe.append([nome, round(v * 1e3, 3), grado, ordine, round(rateo, 4), round(perigeo, 3)])
    with open(os.path.join(OUT_DIR, "controllo_J2_freeflyer.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["caso", "v_apogeo_m_s", "grado", "ordine", "precessione_gradi_giorno", "perigeo_km"])
        w.writerows(righe)


if __name__ == "__main__":
    main()

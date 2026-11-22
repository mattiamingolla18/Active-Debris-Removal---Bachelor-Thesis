"""
phase5_figures.py
=================
Generates all the thesis figures (01-10) into ../figures (PNG, 200 dpi).

No physical parameter is redefined here: every quantity is imported from the
phase modules, so the figures always match the numbers printed by phases 2-4.

  phase2_phasing.py  -> phasing geometry and Delta-V (400 km -> 550 km)
  phase3_deorbit.py  -> deorbit burn, mass budget, ballistic coefficient
  phase4_decay.py    -> US-76 atmosphere model and re-entry propagation

Simplified model: point-mass Earth (no J2), coplanar equatorial orbits,
drag with a US-76 atmosphere co-rotating with the Earth.

Figures:
  01  3D view of the phasing orbit
  02  Rendezvous geometry and closing of the phase angle
  03  Chaser-target relative distance during phasing
  04  Mass and propellant as a function of the operational altitude
  05  Atmospheric density model and drag
  06  Orbital decay: altitude and circularisation over time
  07  Evolution of the re-entry orbit and energy dissipation
  08  Parametric study: ballistic coefficient and perigee altitude
  09  Comparison with FreeFlyer (reads Burn1-3.txt, report_range_rendezvous.Report
      and "Starlink Epoch.txt" from ../freeflyer)
  10  Mission mass budget

Figure labels are in Italian, as in the thesis.
Run:  python phase5_figures.py   (~1 minute)
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra la proiezione 3d)
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUTDIR = os.path.join(HERE, "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import delle fasi precedenti
# ---------------------------------------------------------------------------
# phase4_decay runs the full re-entry propagation when imported (~10 s).
print("Caricamento delle fasi 2-3-4 (la propagazione del rientro richiede ~10 s)...")
_t0 = time.time()

import phase2_phasing as P2   # noqa: E402
import phase3_deorbit as P3   # noqa: E402
import phase4_decay as P4     # noqa: E402

print(f"...fasi caricate in {time.time()-_t0:.1f} s\n")

# ---------------------------------------------------------------------------
# Costanti e grandezze derivate (tutte ereditate, nessun valore riscritto)
# ---------------------------------------------------------------------------
MU = P2.mu                     # km^3/s^2
RE = P2.r_earth                # km
G0 = P2.g                      # m/s^2
ISP = P2.Isp                   # s

H_CHASER = P2.r_chaser - RE    # 400 km, orbita di parcheggio
H_TARGET = P2.r_target - RE    # 550 km, orbita del detrito (Starlink)
H_DEORBIT = P3.r_deorbit - RE  # 200 km, perigeo dell'ellisse di rientro

DV1 = abs(P2.delta_v1_chaser) * 1000.0   # m/s
DV2 = abs(P2.delta_v2_chaser) * 1000.0   # m/s
DV3 = abs(P3.delta_v3) * 1000.0          # m/s

# ---------------------------------------------------------------------------
# Stile grafico
# ---------------------------------------------------------------------------
# Palette categorica pre-validata per daltonismo (ordine fisso, mai ciclato):
# slot 1 blu, slot 2 arancio, slot 3 acqua. Il rosso "critical" e' riservato a
# soglie e punti di progetto e non identifica mai una serie di dati.
C_CHASER   = "#2a78d6"   # blu     - chaser / orbita di parcheggio
C_TARGET   = "#eb6834"   # arancio - target / detrito
C_TRANSFER = "#1baf7a"   # acqua   - orbite di trasferimento
C_LIMIT    = "#d03b3b"   # rosso   - soglie, limiti, punti di progetto
C_INK      = "#0b0b0b"
C_INK2     = "#52514e"
C_MUTED    = "#898781"
C_GRID     = "#e1e0d9"
C_EARTH    = "#9ec5f4"

# Rampa sequenziale a tinta unica (blu chiaro -> blu scuro) per le grandezze
# ordinali: impulso specifico, tempo trascorso durante il decadimento.
RAMPA_BLU = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
CMAP_BLU = LinearSegmentedColormap.from_list("blu_seq", ["#cde2fb", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": C_INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.7,
    "xtick.color": C_INK2,
    "ytick.color": C_INK2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": C_GRID,
    "legend.fontsize": 9.5,
    "font.size": 10.5,
    "lines.linewidth": 2.0,
})


def salva(fig, nome):
    """Salva la figura in figures/<nome>.png e ne stampa il nome."""
    path = os.path.join(OUTDIR, nome + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [ok] {nome}.png")
    return path


# ---------------------------------------------------------------------------
# Utility orbitali
# ---------------------------------------------------------------------------
def kepler_E(M, e, tol=1e-13, itmax=80):
    """Anomalia eccentrica E da anomalia media M (Newton-Raphson)."""
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi) * np.ones_like(M, dtype=float)
    for _ in range(itmax):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def anomalia_vera(t, a, e, mu=MU):
    """Anomalia vera dopo un tempo t dal passaggio al perigeo."""
    n = np.sqrt(mu / a ** 3)
    E = kepler_E(n * np.asarray(t, dtype=float), e)
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))


def dv_hohmann(r1, r2, mu=MU):
    """Delta-V di partenza e di arrivo per un trasferimento di Hohmann [km/s]."""
    a = (r1 + r2) / 2.0
    dv_a = np.abs(np.sqrt(2 * mu / r1 - mu / a) - np.sqrt(mu / r1))
    dv_b = np.abs(np.sqrt(mu / r2) - np.sqrt(2 * mu / r2 - mu / a))
    return dv_a, dv_b


def tsiolkovsky(m0, dv_ms, isp=ISP, g0=G0):
    """Propellente consumato [kg] per un Delta-V in m/s a partire da m0 [kg]."""
    return m0 * (1.0 - np.exp(-np.asarray(dv_ms, dtype=float) / (isp * g0)))


# --- Geometria del phasing --------------------------------------------------
# Il chaser parte con anomalia vera 0 (asse +X), il target con 45 gradi, come
# nella scena FreeFlyer. Dopo il tempo di attesa il chaser ha percorso n_C*wt.
# NB: tutti gli angoli polari vengono riportati in [0, 2*pi) perche' target e
# chaser compiono piu' rivoluzioni complete durante l'attesa.
THETA_BURN = np.mod(P2.n_C * P2.wt, 2 * np.pi)      # angolo polare del Delta-V1
THETA_T0 = np.radians(P2.delta_phi_0)               # angolo polare iniziale target
THETA_T_BURN = np.mod(THETA_T0 + P2.n_T * P2.wt, 2 * np.pi)   # target al Delta-V1
THETA_DV2 = np.mod(THETA_BURN + np.pi, 2 * np.pi)   # angolo polare del Delta-V2
T_RENDEZVOUS = P2.wt + P2.TOF


def stato_chaser(t):
    """Posizione del chaser nel piano [km] durante attesa + trasferimento."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    r = np.empty_like(t)
    ang = np.empty_like(t)

    coast = t <= P2.wt                    # fase di attesa sull'orbita a 400 km
    r[coast] = P2.r_chaser
    ang[coast] = P2.n_C * t[coast]

    tr = ~coast                           # ellisse di trasferimento
    nu = anomalia_vera(t[tr] - P2.wt, P2.a_phas, P2.e_phas)
    r[tr] = P2.a_phas * (1 - P2.e_phas ** 2) / (1 + P2.e_phas * np.cos(nu))
    ang[tr] = THETA_BURN + nu
    return r, ang


def stato_target(t):
    """Posizione del target nel piano [km] (orbita circolare a 550 km)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.full_like(t, P2.r_target), THETA_T0 + P2.n_T * t


# ---------------------------------------------------------------------------
# Modello atmosferico e propagatore del rientro (riusati da phase4_decay)
# ---------------------------------------------------------------------------
densita = np.vectorize(P4.density)       # P4.density e' scalare: qui vettoriale


def dinamica_drag(B_c):
    """Campo vettoriale 2D gravita' + drag per un dato coefficiente balistico."""
    def f(t, s):
        x, y, vx, vy = s
        r = np.hypot(x, y)
        rho = P4.density(r - RE)
        v_atm = np.array([-P4.w_earth * y, P4.w_earth * x])
        v_rel = (np.array([vx, vy]) - v_atm) * 1000.0        # km/s -> m/s
        v_n = np.hypot(v_rel[0], v_rel[1])
        a_d = -0.5 * rho * v_n ** 2 / B_c / 1000.0           # m/s^2 -> km/s^2
        return [vx, vy,
                -MU * x / r ** 3 + a_d * v_rel[0] / v_n,
                -MU * y / r ** 3 + a_d * v_rel[1] / v_n]
    return f


def evento_rientro(t, s):
    return np.hypot(s[0], s[1]) - RE - P4.h_reentry


evento_rientro.terminal = True
evento_rientro.direction = -1


def stato_iniziale(h_apo, h_peri):
    """Stato all'apogeo dell'ellisse h_apo x h_peri [km]."""
    ra, rp = RE + h_apo, RE + h_peri
    a = (ra + rp) / 2.0
    return [ra, 0.0, 0.0, np.sqrt(MU * (2 / ra - 1 / a))]


def tempo_decadimento(h_apo, h_peri, B_c, t_max_giorni=1500):
    """Tempo di rientro [giorni] fino alla quota convenzionale di phase4_decay."""
    sol = solve_ivp(dinamica_drag(B_c), [0, t_max_giorni * 86400],
                    stato_iniziale(h_apo, h_peri), method="DOP853",
                    rtol=1e-7, atol=1e-8, events=evento_rientro)
    if sol.t_events[0].size:
        return sol.t_events[0][0] / 86400.0
    return np.nan


# ---------------------------------------------------------------------------
# Lettura dei file prodotti da FreeFlyer (per la figura di validazione)
# ---------------------------------------------------------------------------
FF_DIR = os.path.join(HERE, "..", "freeflyer")
_MESI = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                     "Aug", "Sep", "Oct", "Nov", "Dec"), start=1)}
_NUM = r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?"

# Origine del "Modified Julian" di FreeFlyer (JD 2430000.0): nei Burn*.txt
# 28849.7326388889 corrisponde a Jan 01 2020 05:35:00.000 UTC.
FF_MJD0 = datetime(1941, 1, 5, 12)
# Spacecraft.Epoch scritto con Report e' in scala TAI: nella simulazione del
# 26/09/2026 il valore 28870.7525459 d corrispondeva all'istante finale mostrato
# da FreeFlyer, Jan 22 2020 06:03:02.967 UTC, cioe' 37 s = TAI - UTC (dal 2017).
TAI_MENO_UTC = 37.0     # s


def trova_file_freeflyer(*nomi_file):
    """Percorso del primo file esistente tra i nomi indicati, nella cartella freeflyer/."""
    for nome in nomi_file:
        path = os.path.join(FF_DIR, nome)
        if os.path.isfile(path):
            return path
    return None


def _leggi(*nomi_file):
    path = trova_file_freeflyer(*nomi_file)
    if path is None:
        return None
    with open(path, "r", errors="ignore") as fh:
        return fh.read()


def leggi_epoca(testo):
    """Epoca UTC contenuta in un testo prodotto da FreeFlyer.

    Accetta la data di calendario 'Mmm DD YYYY HH:MM:SS.sss' (Burn*.txt,
    Spacecraft.EpochText) oppure un report di Spacecraft.Epoch in giorni
    (Modified Julian di FreeFlyer, scala TAI).
    """
    m = re.search(r"([A-Z][a-z]{2}) (\d{1,2}) (\d{4}) (\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", testo)
    if m:
        mese, g, a, hh, mm, ss = m.groups()
        return (datetime(int(a), _MESI[mese], int(g), int(hh), int(mm))
                + timedelta(seconds=float(ss)))
    m = re.search(r"Epoch\s+day\s+(" + _NUM + r")", testo)
    if m:
        return FF_MJD0 + timedelta(days=float(m.group(1)), seconds=-TAI_MENO_UTC)
    return None


def leggi_burn_freeflyer(nome_file):
    """Epoca, stati pre/post manovra, Delta-V [m/s] e masse da un Maneuver Summary File."""
    testo = _leggi(nome_file)
    if testo is None:
        return None
    i_pre, i_post, i_net = (testo.find(k) for k in ("Premaneuver", "Postmaneuver", "Net ICRF"))
    dv = re.search(r"Magnitude:\s+(" + _NUM + r")\s*m/s", testo)
    mp = re.search(r"Mass used:\s+(" + _NUM + r")\s*kg", testo)
    m0 = re.search(r"Starting mass:\s+(" + _NUM + r")\s*kg", testo)
    if min(i_pre, i_post, i_net) < 0 or not (dv and mp and m0):
        return None

    def stato(blocco):
        v = dict(re.findall(r"^\s*(VX|VY|VZ|X|Y|Z)\s+(" + _NUM + r")", blocco, re.M))
        return np.array([float(v[k]) for k in ("X", "Y", "Z", "VX", "VY", "VZ")])

    return {"epoca": leggi_epoca(testo),
            "pre": stato(testo[i_pre:i_post]), "post": stato(testo[i_post:i_net]),
            "dv": float(dv.group(1)), "m_prop": float(mp.group(1)),
            "m0": float(m0.group(1))}


def leggi_report_freeflyer(*nomi_file):
    """(epoca o None, ultimo valore numerico) dal piu' recente dei report indicati."""
    testo = _leggi(*nomi_file)
    if testo is None:
        return None
    numeri = re.findall(_NUM, testo)
    return leggi_epoca(testo), (float(numeri[-1]) if numeri else None)


def configurazione_freeflyer():
    """Epoca iniziale e descrizione del force model letti dal MissionPlan."""
    epoca0, descr = datetime(2020, 1, 1), "configurazione non letta"
    piani = [f for f in os.listdir(FF_DIR) if f.endswith(".MissionPlan")] \
        if os.path.isdir(FF_DIR) else []
    if piani:
        with open(os.path.join(FF_DIR, piani[0]), "r", errors="ignore") as fh:
            xml = fh.read()
        m = re.search(r"<Epoch>([^<]+)</Epoch>", xml)
        epoca0 = leggi_epoca(m.group(1)) if m else epoca0
        fm = xml[xml.find('ForceModel name="Chaser_ForceModel"'):]

        def terzo_valore(tag):          # la Terra e' il 3o corpo della lista FreeFlyer
            m = re.search(r"<" + tag + r"[^>]*>(?:\s*<value>(\d+)</value>){3}", fm)
            return m.group(1) if m else None

        # PlanetFieldType: 0 punto materiale (grado/ordine ignorati), 1 solo zonali,
        # 2 zonali + tesserali, 3 zonali + tesserali + maree solide
        tipo, grado, ordine = (terzo_valore(t) for t in
                               ("PlanetFieldType", "PlanetFieldDegree", "PlanetFieldOrder"))
        atm = re.search(r"<AtmosphericDensityModel>(\d)<", fm)
        passo = re.search(r'name="Chaser_Propagator".*?<StepSize[^>]*>([\d.]+)<', xml, re.S)
        modelli = {"1": "Analytic", "2": "Harris-Priester", "3": "Jacchia-Roberts",
                   "4": "MSIS-2000", "5": "MSIS-90", "6": "MSIS-86"}
        gravita = {"0": "Terra puntiforme",
                   "1": f"gravita' solo zonali fino al grado {grado}",
                   "2": f"gravita' {grado}x{ordine}",
                   "3": f"gravita' {grado}x{ordine} + maree solide"}.get(tipo, "gravita' ?")
        descr = (f"{gravita}, atmosfera {modelli.get(atm.group(1), '?') if atm else '?'}, "
                 f"passo {passo.group(1) if passo else '?'} s")
    return epoca0, descr


def apsidi(stato):
    """Quote di apogeo e perigeo [km] dell'orbita kepleriana osculatrice."""
    r, v = stato[:3], stato[3:]
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    a = 1.0 / (2.0 / rn - vn ** 2 / MU)
    e = np.linalg.norm(((vn ** 2 - MU / rn) * r - np.dot(r, v) * v) / MU)
    return a * (1 + e) - RE, a * (1 - e) - RE


FF_BURNS = {n: leggi_burn_freeflyer(f"Burn{n}.txt") for n in (1, 2, 3)}
# report opzionali: da aggiungere al MissionPlan con "Report ... to <file>;"
FF_RANGE = leggi_report_freeflyer("report_range_rendezvous.txt",
                                  "report_range_rendezvous.Report")
FF_RIENTRO = leggi_report_freeflyer("report_rientro.txt", "Starlink Epoch.txt")
FF_EPOCA0, FF_CONFIG = configurazione_freeflyer()


# ===========================================================================
# FIGURA 1 - Visualizzazione 3D dell'orbita di phasing
# ===========================================================================
def figura_1_phasing_3d():
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0],
                          wspace=0.02, hspace=0.34)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    axz = fig.add_subplot(gs[0, 1])
    axp = fig.add_subplot(gs[1, 1], projection="polar")

    # ----------------------------------------------------------------- 3D --
    u = np.linspace(0, 2 * np.pi, 160)
    v = np.linspace(0, np.pi, 80)
    ax.plot_surface(RE * np.outer(np.cos(u), np.sin(v)),
                    RE * np.outer(np.sin(u), np.sin(v)),
                    RE * np.outer(np.ones_like(u), np.cos(v)),
                    color=C_EARTH, alpha=0.45, linewidth=0,
                    antialiased=True, shade=True, zorder=0)
    # reticolo geografico: da' profondita' alla sfera e rende leggibile il 3D
    for lon in np.arange(0, 360, 30):
        th = np.radians(lon)
        ax.plot(RE * np.cos(th) * np.sin(v), RE * np.sin(th) * np.sin(v),
                RE * np.cos(v), color="white", lw=0.4, alpha=0.55)
    for lat in np.arange(-60, 61, 30):
        ph = np.radians(lat)
        ax.plot(RE * np.cos(ph) * np.cos(u), RE * np.cos(ph) * np.sin(u),
                RE * np.sin(ph) * np.ones_like(u), color="white", lw=0.4, alpha=0.55)

    # orbite: i = 0 deg, complanari ed equatoriali come nel MissionPlan FreeFlyer
    th = np.linspace(0, 2 * np.pi, 720)
    ax.plot(P2.r_chaser * np.cos(th), P2.r_chaser * np.sin(th), 0 * th,
            color=C_CHASER, lw=1.5, ls="--", alpha=0.85,
            label=f"Orbita di parcheggio Chaser - {H_CHASER:.0f} km")
    ax.plot(P2.r_target * np.cos(th), P2.r_target * np.sin(th), 0 * th,
            color=C_TARGET, lw=1.5, ls="--", alpha=0.85,
            label=f"Orbita del Target (Starlink) - {H_TARGET:.0f} km")

    th_wait = np.linspace(0, THETA_BURN, 400)
    ax.plot(P2.r_chaser * np.cos(th_wait), P2.r_chaser * np.sin(th_wait), 0 * th_wait,
            color=C_CHASER, lw=3.0, solid_capstyle="round",
            label=f"Attesa in orbita bassa - {P2.wt/3600:.2f} h "
                  f"({P2.wt/P2.T_chaser:.2f} rivoluzioni)")

    nu = np.linspace(0, np.pi, 500)
    r_tr = P2.a_phas * (1 - P2.e_phas ** 2) / (1 + P2.e_phas * np.cos(nu))
    ax.plot(r_tr * np.cos(THETA_BURN + nu), r_tr * np.sin(THETA_BURN + nu), 0 * nu,
            color=C_TRANSFER, lw=3.4, solid_capstyle="round",
            label=f"Ellisse di Hohmann - a = {P2.a_phas:.1f} km, "
                  f"e = {P2.e_phas:.4f}, TOF {P2.TOF/60:.1f} min")

    p1 = (P2.r_chaser * np.cos(THETA_BURN), P2.r_chaser * np.sin(THETA_BURN))
    p2 = (P2.r_target * np.cos(THETA_DV2), P2.r_target * np.sin(THETA_DV2))

    ax.scatter(P2.r_chaser, 0, 0, s=80, color=C_CHASER, edgecolor="white",
               lw=1.2, depthshade=False, zorder=6,
               label="Chaser a t = 0 ($\\theta$ = 0$\\degree$)")
    ax.scatter(P2.r_target * np.cos(THETA_T0), P2.r_target * np.sin(THETA_T0), 0,
               s=80, color=C_TARGET, edgecolor="white", lw=1.2,
               depthshade=False, zorder=6,
               label=f"Target a t = 0 ($\\theta$ = {P2.delta_phi_0:.0f}$\\degree$)")
    ax.scatter(*p1, 0, s=170, marker="^", color=C_LIMIT, edgecolor="white",
               lw=1.2, depthshade=False, zorder=7,
               label=f"$\\Delta V_1$ = {DV1:.2f} m/s (prograda, al perigeo)")
    ax.scatter(*p2, 0, s=170, marker="v", color=C_LIMIT, edgecolor="white",
               lw=1.2, depthshade=False, zorder=7,
               label=f"$\\Delta V_2$ = {DV2:.2f} m/s (circolarizzazione, all'apogeo)")

    ax.text(p1[0] * 1.14, p1[1] * 1.14, -900, "$\\Delta V_1$", color=C_LIMIT,
            fontsize=13, fontweight="bold", ha="center")
    ax.text(p2[0] * 1.14, p2[1] * 1.14, 900, "$\\Delta V_2$", color=C_LIMIT,
            fontsize=13, fontweight="bold", ha="center")

    # vettori Delta-V, tangenti e progradi (lunghezza non in scala)
    L = 1600.0
    for (px, py), th_p in ((p1, THETA_BURN), (p2, THETA_DV2)):
        ax.quiver(px, py, 0, -np.sin(th_p) * L, np.cos(th_p) * L, 0,
                  color=C_LIMIT, lw=2.2, arrow_length_ratio=0.3)

    lim = P2.r_target * 1.10
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))      # scala isotropa: geometria non deformata
    ax.set_xlabel("X ICRF [km]", labelpad=14)
    ax.set_ylabel("Y ICRF [km]", labelpad=14)
    ax.set_zlabel("Z ICRF [km]", labelpad=6)
    ax.view_init(elev=26, azim=-60)
    ax.tick_params(labelsize=8.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.03)
    ax.set_title("(a) Vista tridimensionale in scala reale", pad=2)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, 0.99), fontsize=9)

    # ------------------------------------------------- quota nel tempo (b) --
    tt = np.linspace(0, T_RENDEZVOUS, 4000)
    r_c, _ = stato_chaser(tt)
    axz.plot(tt / 60, r_c - RE, color=C_TRANSFER, lw=2.4)
    axz.axhline(H_CHASER, color=C_CHASER, ls="--", lw=1.4)
    axz.axhline(H_TARGET, color=C_TARGET, ls="--", lw=1.4)
    axz.axvline(P2.wt / 60, color=C_LIMIT, ls=":", lw=1.5)
    axz.axvline(T_RENDEZVOUS / 60, color=C_LIMIT, ls=":", lw=1.5)
    axz.text(P2.wt / 60 - 8, 470, "$\\Delta V_1$", color=C_LIMIT, fontsize=11,
             fontweight="bold", ha="right")
    axz.text(T_RENDEZVOUS / 60 + 6, 470, "$\\Delta V_2$", color=C_LIMIT,
             fontsize=11, fontweight="bold")
    axz.text(20, H_CHASER + 8, f"{H_CHASER:.0f} km", color=C_CHASER, fontsize=9)
    axz.text(20, H_TARGET - 22, f"{H_TARGET:.0f} km", color=C_TARGET, fontsize=9)
    axz.set_xlim(0, T_RENDEZVOUS / 60 * 1.06)
    axz.set_ylim(385, 575)
    axz.set_xlabel("Tempo [min]")
    axz.set_ylabel("Quota [km]")
    axz.set_title("(b) Profilo di quota del Chaser", fontsize=11)
    axz.grid(True, ls=":", alpha=0.6)

    # ------------------------------------ vista polare in coordinate quota --
    # Le tre orbite distano solo 150 km su un raggio di ~6900 km: in scala
    # reale (pannello a) risultano quasi sovrapposte. Qui l'asse radiale e' la
    # QUOTA, non il raggio, cosi' la manovra diventa leggibile.
    axp.plot(th, np.full_like(th, H_CHASER), color=C_CHASER, ls="--", lw=1.6)
    axp.plot(th, np.full_like(th, H_TARGET), color=C_TARGET, ls="--", lw=1.6)
    axp.plot(THETA_BURN + nu, r_tr - RE, color=C_TRANSFER, lw=3.0)
    axp.plot(th_wait, np.full_like(th_wait, H_CHASER), color=C_CHASER, lw=3.0)
    axp.plot(THETA_BURN, H_CHASER, "^", ms=12, color=C_LIMIT, mec="white",
             mew=1.2, zorder=6)
    axp.plot(THETA_DV2, H_TARGET, "v", ms=12, color=C_LIMIT, mec="white",
             mew=1.2, zorder=6)
    axp.set_rlim(300, 580)
    axp.set_rticks([350, 400, 450, 500, 550])
    axp.set_rlabel_position(112)
    axp.tick_params(labelsize=8.5)
    axp.grid(True, ls=":", alpha=0.7)
    axp.set_title("(c) Stessa manovra con asse radiale = quota [km]",
                  fontsize=11, pad=14)

    fig.suptitle("Fase 2 - Orbita di phasing: trasferimento di Hohmann "
                 f"{H_CHASER:.0f} km $\\to$ {H_TARGET:.0f} km, orbite equatoriali "
                 f"complanari (i = 0$\\degree$)",
                 fontsize=14, fontweight="bold", y=0.985)
    salva(fig, "01_phasing_orbita_3D")


# ===========================================================================
# FIGURA 2 - Geometria del phasing nel piano orbitale
# ===========================================================================
def figura_2_geometria_phasing():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.5, 6.6))

    # --- (a) vista dall'alto ---------------------------------------------
    th = np.linspace(0, 2 * np.pi, 720)
    axa.add_patch(plt.Circle((0, 0), RE, color=C_EARTH, alpha=0.55, zorder=0))
    axa.plot(P2.r_chaser * np.cos(th), P2.r_chaser * np.sin(th),
             color=C_CHASER, ls="--", lw=1.5,
             label=f"Orbita Chaser {H_CHASER:.0f} km")
    axa.plot(P2.r_target * np.cos(th), P2.r_target * np.sin(th),
             color=C_TARGET, ls="--", lw=1.5,
             label=f"Orbita Target {H_TARGET:.0f} km")

    nu = np.linspace(0, np.pi, 400)
    r_tr = P2.a_phas * (1 - P2.e_phas ** 2) / (1 + P2.e_phas * np.cos(nu))
    axa.plot(r_tr * np.cos(THETA_BURN + nu), r_tr * np.sin(THETA_BURN + nu),
             color=C_TRANSFER, lw=3.2, label="Ellisse di trasferimento")
    th_wait = np.linspace(0, THETA_BURN, 400)
    axa.plot(P2.r_chaser * np.cos(th_wait), P2.r_chaser * np.sin(th_wait),
             color=C_CHASER, lw=3.0,
             label=f"Arco di attesa (ultimi {THETA_BURN/(2*np.pi):.2f} giri)")

    # archi degli angoli notevoli, tracciati appena fuori dall'orbita target
    r_a = P2.r_target * 1.10
    arc0 = np.linspace(0, THETA_T0, 120)
    axa.plot(r_a * np.cos(arc0), r_a * np.sin(arc0), color=C_MUTED, lw=2.0)
    axa.annotate(f"$\\Delta\\varphi_0$ = {P2.delta_phi_0:.0f}$\\degree$",
                 xy=(r_a * np.cos(THETA_T0 / 2), r_a * np.sin(THETA_T0 / 2)),
                 xytext=(P2.r_target * 1.05, P2.r_target * 0.92),
                 color=C_INK, fontsize=12, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3))

    arc1 = np.linspace(THETA_BURN, THETA_T_BURN, 120)
    axa.plot(r_a * np.cos(arc1), r_a * np.sin(arc1), color=C_LIMIT, lw=2.4)
    m1 = (THETA_BURN + THETA_T_BURN) / 2
    axa.annotate(f"$\\varphi$ = {np.degrees(P2.phi):.1f}$\\degree$\n"
                 f"(lead angle al $\\Delta V_1$)",
                 xy=(r_a * np.cos(m1), r_a * np.sin(m1)),
                 xytext=(-P2.r_target * 0.62, -P2.r_target * 1.28),
                 color=C_LIMIT, fontsize=10.5, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.3))

    # marcatori identificati dalla legenda: evita etichette sovrapposte, dato
    # che il punto di Delta-V2 e la posizione iniziale del target distano 2 gradi
    punti = ((0.0, P2.r_chaser, "o", C_CHASER, 10, "Chaser a t = 0"),
             (THETA_T0, P2.r_target, "o", C_TARGET, 10, "Target a t = 0"),
             (THETA_BURN, P2.r_chaser, "^", C_LIMIT, 13,
              f"$\\Delta V_1$ dopo {P2.wt/3600:.2f} h di attesa"),
             (THETA_T_BURN, P2.r_target, "s", C_TARGET, 9,
              "Target all'istante del $\\Delta V_1$"),
             (THETA_DV2, P2.r_target, "v", C_LIMIT, 13,
              "$\\Delta V_2$ - rendez-vous"))
    for ang, r, mk, col, ms, lab in punti:
        axa.plot(r * np.cos(ang), r * np.sin(ang), mk, ms=ms, color=col,
                 mec="white", mew=1.4, zorder=6, label=lab, ls="none")

    lim = P2.r_target * 1.42
    axa.set_xlim(-lim, lim)
    axa.set_ylim(-lim, lim)
    axa.set_aspect("equal")
    axa.set_xlabel("X [km]")
    axa.set_ylabel("Y [km]")
    axa.set_title("(a) Geometria del rendez-vous nel piano equatoriale")
    axa.legend(loc="upper left", fontsize=8.6)
    axa.grid(True, ls=":", alpha=0.45)

    # --- (b) angolo di fase nel tempo ------------------------------------
    t = np.linspace(0, T_RENDEZVOUS, 6000)
    _, a_c = stato_chaser(t)
    _, a_t = stato_target(t)
    dphi = np.degrees(np.mod(a_t - a_c + np.pi, 2 * np.pi) - np.pi)

    axb.axvspan(0, P2.wt / 3600, color=C_CHASER, alpha=0.06)
    axb.axvspan(P2.wt / 3600, T_RENDEZVOUS / 3600, color=C_TRANSFER, alpha=0.12)
    axb.plot(t / 3600, dphi, color=C_TRANSFER, lw=2.6,
             label="Angolo di fase $\\Delta\\varphi(t)$")
    axb.axhline(0, color=C_LIMIT, ls="--", lw=1.5,
                label="Rendez-vous ($\\Delta\\varphi$ = 0)")
    axb.axhline(np.degrees(P2.phi), color=C_MUTED, ls=":", lw=1.5,
                label=f"Lead angle $\\varphi$ = {np.degrees(P2.phi):.1f}$\\degree$")
    for tv, lab in ((P2.wt / 3600, "$\\Delta V_1$"),
                    (T_RENDEZVOUS / 3600, "$\\Delta V_2$")):
        axb.axvline(tv, color=C_LIMIT, ls="-.", lw=1.4)
        axb.text(tv, 46.5, lab, color=C_LIMIT, fontsize=11, fontweight="bold",
                 ha="center", va="bottom")
    axb.text(P2.wt / 7200, 41, "attesa in orbita bassa\n(il Chaser recupera fase)",
             color=C_CHASER, ha="center", fontsize=10, fontweight="bold")
    axb.text((P2.wt + T_RENDEZVOUS) / 7200, 24, "trasferimento",
             color=C_TRANSFER, ha="center", fontsize=10, fontweight="bold",
             rotation=90)

    axb.set_xlim(0, T_RENDEZVOUS / 3600 * 1.05)
    axb.set_ylim(-3, 50)
    axb.set_xlabel("Tempo dalla condizione iniziale [h]")
    axb.set_ylabel("Angolo di fase Target $-$ Chaser [$\\degree$]")
    axb.set_title(f"(b) Chiusura dell'angolo di fase: attesa "
                  f"{int(P2.wt)//3600} h {int(P2.wt%3600)//60} min, "
                  f"TOF {P2.TOF/60:.1f} min")
    axb.legend(loc="center left")
    axb.grid(True, ls=":", alpha=0.6)

    fig.suptitle("Fase 2 - Strategia di phasing: attesa in orbita bassa e "
                 "trasferimento di Hohmann", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    salva(fig, "02_phasing_geometria")


# ===========================================================================
# FIGURA 3 - Distanza relativa Chaser-Target durante il phasing
# ===========================================================================
def figura_3_range():
    t = np.linspace(0, T_RENDEZVOUS, 8000)
    r_c, a_c = stato_chaser(t)
    r_t, a_t = stato_target(t)
    d = np.sqrt(r_c ** 2 + r_t ** 2 - 2 * r_c * r_t * np.cos(a_t - a_c))

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.axvspan(0, P2.wt / 3600, color=C_CHASER, alpha=0.06)
    ax.axvspan(P2.wt / 3600, T_RENDEZVOUS / 3600, color=C_TRANSFER, alpha=0.12)
    ax.plot(t / 3600, d, color=C_TRANSFER, lw=2.6)

    for tv, lab in ((P2.wt / 3600, "$\\Delta V_1$"),
                    (T_RENDEZVOUS / 3600, "$\\Delta V_2$")):
        ax.axvline(tv, color=C_LIMIT, ls="-.", lw=1.4)
        ax.text(tv, d.max() * 1.015, lab, color=C_LIMIT, fontsize=11,
                fontweight="bold", ha="center", va="bottom")

    ax.plot(T_RENDEZVOUS / 3600, d[-1], "o", ms=11, color=C_LIMIT,
            mec="white", mew=1.5, zorder=6)
    ax.annotate(f"range residuo al $\\Delta V_2$: {d[-1]:.1f} km",
                xy=(T_RENDEZVOUS / 3600, d[-1]),
                xytext=(T_RENDEZVOUS / 3600 * 0.60, d.max() * 0.16),
                color=C_LIMIT, fontweight="bold", fontsize=10.5, ha="right",
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.4))
    i_wt = np.searchsorted(t, P2.wt)
    v_chiusura = (d[0] - d[i_wt]) / (t[i_wt] / 60)     # km/min, media sull'attesa
    ax.text(P2.wt / 7200, d.max() * 0.90,
            f"attesa in orbita di parcheggio\n(chiusura media di {v_chiusura:.1f} km/min)",
            color=C_CHASER, ha="center", fontsize=10.5, fontweight="bold")

    ax.set_xlim(0, T_RENDEZVOUS / 3600 * 1.02)
    ax.set_ylim(-d.max() * 0.03, d.max() * 1.10)
    ax.set_xlabel("Tempo dalla condizione iniziale [h]")
    ax.set_ylabel("Distanza Chaser $-$ Target [km]")
    ax.set_title("Fase 2 - Evoluzione della distanza relativa Chaser / Starlink\n"
                 "il phasing porta i due veicoli in condizione di rendez-vous "
                 "al secondo impulso")
    ax.grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    salva(fig, "03_phasing_distanza_relativa")


# ===========================================================================
# FIGURA 4 - Massa e propellente al variare della quota  [richiesta esplicita]
# ===========================================================================
def figura_4_massa_vs_quota():
    h = np.linspace(420.0, 1200.0, 400)       # quota del detrito da rimuovere
    r2 = RE + h

    # (1) Phasing: 400 km -> h, massa iniziale del chaser 400 kg
    dva, dvb = dv_hohmann(P2.r_chaser, r2)
    dv_phas = (dva + dvb) * 1000.0
    mp_phas = tsiolkovsky(P2.mass_chaser, dv_phas)
    m_dopo_phas = P2.mass_chaser - mp_phas

    # (2) Deorbiting: da h fino a perigeo 200 km, con il detrito agganciato
    dv_deo = np.abs(np.sqrt(MU / r2) -
                    np.sqrt(2 * MU / r2 - MU / ((r2 + P3.r_deorbit) / 2))) * 1000.0
    m_sistema = m_dopo_phas + P2.starlink_mass
    mp_deo = tsiolkovsky(m_sistema, dv_deo)

    mp_tot = mp_phas + mp_deo
    m_finale = P2.mass_chaser - mp_tot

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14.5, 10))

    # --- (a) Delta-V ------------------------------------------------------
    a1.plot(h, (dva + dvb) * 1000, color=C_CHASER, lw=2.4,
            label="$\\Delta V_1 + \\Delta V_2$ (phasing)")
    a1.plot(h, dv_deo, color=C_TARGET, lw=2.4, label="$\\Delta V_3$ (deorbiting)")
    a1.plot(h, (dva + dvb) * 1000 + dv_deo, color=C_TRANSFER, lw=2.6,
            label="$\\Delta V$ totale di missione")
    a1.axvline(H_TARGET, color=C_LIMIT, ls="--", lw=1.5)
    a1.plot(H_TARGET, DV1 + DV2 + DV3, "*", ms=18, color=C_LIMIT,
            mec="white", mew=1.2, zorder=6)
    a1.annotate(f"missione a {H_TARGET:.0f} km\n{DV1+DV2+DV3:.1f} m/s",
                xy=(H_TARGET, DV1 + DV2 + DV3), xytext=(660, 470),
                color=C_LIMIT, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.3))
    a1.set_xlabel("Quota del detrito [km]")
    a1.set_ylabel("$\\Delta V$ [m/s]")
    a1.set_title("(a) Costo in $\\Delta V$ al variare della quota del detrito")
    a1.legend(loc="upper left")
    a1.grid(True, ls=":", alpha=0.6)

    # --- (b) propellente --------------------------------------------------
    a2.fill_between(h, 0, mp_tot, color=C_TRANSFER, alpha=0.07)
    a2.plot(h, mp_phas, color=C_CHASER, lw=2.4, label="Propellente phasing")
    a2.plot(h, mp_deo, color=C_TARGET, lw=2.4, label="Propellente deorbiting")
    a2.plot(h, mp_tot, color=C_TRANSFER, lw=2.6, label="Propellente totale")
    a2.axvline(H_TARGET, color=C_LIMIT, ls="--", lw=1.5)
    a2.plot(H_TARGET, P3.total_propellant_burned, "*", ms=18, color=C_LIMIT,
            mec="white", mew=1.2, zorder=6)
    a2.annotate(f"missione a {H_TARGET:.0f} km\n"
                f"{P3.total_propellant_burned:.2f} kg",
                xy=(H_TARGET, P3.total_propellant_burned), xytext=(660, 76),
                color=C_LIMIT, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.3))
    a2.set_xlabel("Quota del detrito [km]")
    a2.set_ylabel("Massa di propellente [kg]")
    a2.set_title(f"(b) Propellente consumato ($I_{{sp}}$ = {ISP:.0f} s, "
                 f"$m_0$ = {P2.mass_chaser:.0f} kg)")
    a2.legend(loc="upper left")
    a2.grid(True, ls=":", alpha=0.6)

    # --- (c) massa a secco residua ---------------------------------------
    a3.fill_between(h, m_finale, P2.mass_chaser, color=C_TARGET, alpha=0.16,
                    label="Propellente speso")
    a3.plot(h, m_finale, color=C_CHASER, lw=2.6,
            label="Massa del Chaser a fine missione")
    a3.axhline(P2.mass_chaser, color=C_MUTED, ls=":", lw=1.5)
    a3.text(1195, P2.mass_chaser - 2.5,
            f"massa al lancio {P2.mass_chaser:.0f} kg", color=C_INK2,
            fontsize=9.5, ha="right", va="top")
    a3.axvline(H_TARGET, color=C_LIMIT, ls="--", lw=1.5)
    a3.plot(H_TARGET, P3.mass_chaser_after_burned, "*", ms=18, color=C_LIMIT,
            mec="white", mew=1.2, zorder=6)
    a3.annotate(f"{P3.mass_chaser_after_burned:.2f} kg",
                xy=(H_TARGET, P3.mass_chaser_after_burned), xytext=(700, 380),
                color=C_LIMIT, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.3))
    a3.set_xlabel("Quota del detrito [km]")
    a3.set_ylabel("Massa del Chaser [kg]")
    a3.set_title("(c) Bilancio di massa del Chaser")
    a3.legend(loc="lower left")
    a3.grid(True, ls=":", alpha=0.6)

    # --- (d) sensibilita' all'impulso specifico ---------------------------
    # I_sp e' una grandezza ordinale: rampa a tinta unica chiaro -> scuro.
    isp_list = [200, 250, 300, 350]
    for isp, col in zip(isp_list, RAMPA_BLU):
        mp_p = tsiolkovsky(P2.mass_chaser, dv_phas, isp=isp)
        mp_d = tsiolkovsky(P2.mass_chaser - mp_p + P2.starlink_mass, dv_deo, isp=isp)
        a4.plot(h, mp_p + mp_d, color=col, lw=2.4,
                label=f"$I_{{sp}}$ = {isp} s" + ("  (progetto)" if isp == ISP else ""))
    a4.axvline(H_TARGET, color=C_LIMIT, ls="--", lw=1.4)
    a4.plot(H_TARGET, P3.total_propellant_burned, "*", ms=18, color=C_LIMIT,
            mec="white", mew=1.2, zorder=6)
    a4.set_xlabel("Quota del detrito [km]")
    a4.set_ylabel("Propellente totale [kg]")
    a4.set_title("(d) Sensibilita' del budget di propellente all'$I_{sp}$")
    a4.legend(loc="upper left")
    a4.grid(True, ls=":", alpha=0.6)

    fig.suptitle("Variazione di massa e propellente al variare della quota operativa",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    salva(fig, "04_massa_propellente_vs_quota")


# ===========================================================================
# FIGURA 5 - Densita' atmosferica  [grafico gia' realizzato, mantenuto]
# ===========================================================================
def figura_5_densita():
    h = np.linspace(0, 1000, 2500)
    rho = densita(h)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 6.6))

    # --- (a) modello di densita' -----------------------------------------
    a1.semilogx(rho, h, color=C_CHASER, lw=2.4,
                label="US Standard Atmosphere 1976\n(esponenziale a tratti)")
    a1.semilogx(P4.rho_atm, P4.h_atm, "o", ms=5, color=C_TARGET,
                mec="white", mew=0.8, label="Punti tabellati (Curtis)")

    quote = [(H_DEORBIT, "perigeo di rientro"), (H_CHASER, "orbita Chaser"),
             (H_TARGET, "orbita Target"), (P4.h_reentry, "quota di rientro")]
    for q, lab in quote:
        rq = float(densita(q))
        a1.plot(rq, q, "s", ms=8, color=C_LIMIT, mec="white", mew=1.1, zorder=6)
        a1.annotate(f"{lab}\n{q:.0f} km - $\\rho$ = {rq:.2e} kg/m$^3$",
                    xy=(rq, q), xytext=(rq * 55, q + 22),
                    fontsize=8.6, color=C_LIMIT, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.0))

    a1.set_xlabel("Densita' atmosferica $\\rho$ [kg/m$^3$]")
    a1.set_ylabel("Quota geometrica [km]")
    a1.set_title("(a) Profilo di densita' atmosferica")
    a1.set_ylim(0, 1000)
    a1.set_xlim(1e-15, 5)
    a1.legend(loc="lower left")
    a1.grid(True, which="both", ls=":", alpha=0.55)

    # --- (b) decelerazione da drag ---------------------------------------
    h2 = np.linspace(100, 700, 1500)
    v = np.sqrt(MU / (RE + h2)) * 1000.0            # velocita' circolare [m/s]
    a_drag = 0.5 * densita(h2) * v ** 2 / P3.B_c    # m/s^2
    a_grav = MU * 1e9 / ((RE + h2) * 1e3) ** 2      # m/s^2

    a2.semilogy(h2, a_grav, color=C_CHASER, lw=2.0, ls="--",
                label="Accelerazione gravitazionale")
    a2.semilogy(h2, a_drag, color=C_TARGET, lw=2.4,
                label=f"Decelerazione da drag ($B_C$ = {P3.B_c:.1f} kg/m$^2$)")
    for q, col in ((H_TARGET, C_TARGET), (H_CHASER, C_CHASER), (H_DEORBIT, C_LIMIT)):
        a2.axvline(q, color=col, ls=":", lw=1.3, alpha=0.85)
        a2.text(q + 7, 2e-3, f"{q:.0f} km", rotation=90, color=col,
                fontsize=9, fontweight="bold", va="bottom")
    a2.set_ylim(1e-9, 1e2)
    a2.set_xlabel("Quota [km]")
    a2.set_ylabel("Accelerazione [m/s$^2$]")
    a2.set_title("(b) Peso relativo del drag sul sistema Chaser + Target")
    a2.legend(loc="lower left")
    a2.grid(True, which="both", ls=":", alpha=0.55)

    fig.suptitle("Modello atmosferico impiegato nella propagazione del rientro",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    salva(fig, "05_densita_atmosferica")


# ===========================================================================
# FIGURA 6 - Decadimento orbitale: quota nel tempo
# ===========================================================================
def _inviluppo(t, h):
    """Estrae apogei e perigei locali dalla storia di quota."""
    d = np.diff(h)
    imax = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    imin = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return (t[imax], h[imax]), (t[imin], h[imin])


def figura_6_decadimento():
    t_d = P4.solution.t / 86400.0
    h = P4.h
    (t_apo, h_apo), (t_per, h_per) = _inviluppo(t_d, h)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 9.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})

    a1.plot(t_d, h, color=C_MUTED, lw=0.25, alpha=0.28,
            label="Quota istantanea (oscillazione apogeo-perigeo)")
    a1.plot(t_apo, h_apo, color=C_TARGET, lw=2.6, label="Apogeo")
    a1.plot(t_per, h_per, color=C_CHASER, lw=2.6, label="Perigeo")
    a1.axhline(P4.h_reentry, color=C_LIMIT, ls="--", lw=1.6,
               label=f"Rientro convenzionale ({P4.h_reentry:.0f} km)")

    if P4.solution.t_events[0].size:
        t_dec = P4.solution.t_events[0][0] / 86400.0
        a1.axvline(t_dec, color=C_LIMIT, ls=":", lw=1.6)
        a1.annotate(f"rientro dopo {t_dec:.1f} giorni\n"
                    f"({P4.n_rev:.0f} rivoluzioni)",
                    xy=(t_dec, P4.h_reentry + 4),
                    xytext=(t_dec * 0.72, 122),
                    color=C_LIMIT, fontweight="bold", fontsize=10.5, ha="right",
                    arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.4))

    a1.set_ylim(80, 580)
    a1.set_ylabel("Quota [km]")
    a1.set_title("Fase 4 - Decadimento orbitale del sistema Chaser + Starlink\n"
                 f"orbita iniziale {H_TARGET:.0f} x {H_DEORBIT:.0f} km, "
                 f"$B_C$ = {P3.B_c:.1f} kg/m$^2$")
    a1.legend(loc="upper right")
    a1.grid(True, ls=":", alpha=0.6)

    # circolarizzazione progressiva dell'orbita
    n = min(len(h_apo), len(h_per))
    if n > 2:
        ecc = h_apo[:n] - h_per[:n]
        a2.fill_between(t_apo[:n], 0, ecc, color=C_TRANSFER, alpha=0.12)
        a2.plot(t_apo[:n], ecc, color=C_TRANSFER, lw=2.6,
                label="Differenza apogeo $-$ perigeo")
    a2.set_xlabel("Tempo dal burn di deorbiting [giorni]")
    a2.set_ylabel("$h_{apo} - h_{peri}$ [km]")
    a2.set_title("Circolarizzazione progressiva: il drag agisce soprattutto al perigeo",
                 fontsize=11)
    a2.legend(loc="upper right")
    a2.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    salva(fig, "06_decadimento_quota")


# ===========================================================================
# FIGURA 7 - Evoluzione dell'orbita di rientro e dissipazione di energia
# ===========================================================================
def figura_7_evoluzione_orbita():
    x, y = P4.x, P4.y
    r = np.hypot(x, y)
    v = np.hypot(P4.solution.y[2], P4.solution.y[3])
    t_d = P4.solution.t / 86400.0
    dt = P4.solution.t[1] - P4.solution.t[0]

    fig = plt.figure(figsize=(14.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.24)
    axp = fig.add_subplot(gs[0, 0], projection="polar")
    axe = fig.add_subplot(gs[0, 1])

    # --- (a) alcune rivoluzioni campione in coordinate (angolo, quota) -----
    # In scala reale l'orbita e' quasi una circonferenza: usando la QUOTA come
    # asse radiale si vede il progressivo abbassamento e la circolarizzazione.
    n_giri = 9
    t_camp = np.linspace(0, P4.solution.t[-1] * 0.995, n_giri)
    norm = Normalize(0, t_d[-1])
    for tk in t_camp:
        i0 = int(tk / dt)
        a_loc = 1.0 / (2.0 / r[i0] - v[i0] ** 2 / MU)      # vis-viva
        # +3 campioni: chiude la rivoluzione senza lasciare buchi visibili
        n_s = int(2 * np.pi * np.sqrt(a_loc ** 3 / MU) / dt) + 3
        i1 = min(i0 + n_s, len(x) - 1)
        axp.plot(np.arctan2(y[i0:i1], x[i0:i1]), r[i0:i1] - RE,
                 color=CMAP_BLU(norm(t_d[i0])), lw=1.9)
    axp.plot(np.linspace(0, 2 * np.pi, 400),
             np.full(400, P4.h_reentry), color=C_LIMIT, ls="--", lw=1.6)
    axp.set_rlim(0, 580)
    axp.set_rticks([100, 200, 300, 400, 500])
    axp.set_rlabel_position(118)
    axp.tick_params(labelsize=8.5)
    axp.grid(True, ls=":", alpha=0.7)
    axp.set_title("(a) Evoluzione dell'orbita durante il rientro\n"
                  "asse radiale = quota [km], 9 rivoluzioni campione",
                  fontsize=11.5, pad=18)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP_BLU), ax=axp,
                      pad=0.11, shrink=0.72)
    cb.set_label("Tempo dal burn di deorbiting [giorni]", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    axp.text(np.radians(200), 40, f"rientro {P4.h_reentry:.0f} km",
             color=C_LIMIT, fontsize=9, fontweight="bold", ha="center")

    # --- (b) energia specifica e semiasse maggiore ------------------------
    energia = v ** 2 / 2.0 - MU / r                   # km^2/s^2

    # media mobile su una rivoluzione: isola la deriva secolare dall'oscillazione
    # entro il singolo giro (per un'orbita kepleriana l'energia sarebbe costante,
    # quindi la curva istantanea e' di fatto sovrapposta alla media).
    w = max(1, int(P4.T_orbit / dt))
    e_med = np.convolve(energia, np.ones(w) / w, mode="valid")
    t_med = t_d[w - 1:]
    # semiasse maggiore dall'energia MEDIATA (eps = -mu/2a), coerente con e_med:
    # il valore istantaneo dell'ultimo campione, gia' dentro l'atmosfera densa,
    # non rappresenta piu' un'orbita
    a_med = -MU / (2.0 * e_med) - RE                 # quota equivalente [km]
    axe.plot(t_med, e_med, color=C_CHASER, lw=2.8)
    axe.fill_between(t_med, e_med[0], e_med, color=C_CHASER, alpha=0.10)
    axe.axhline(e_med[0], color=C_MUTED, ls=":", lw=1.4)
    axe.set_xlabel("Tempo dal burn di deorbiting [giorni]")
    axe.set_ylabel("Energia specifica $\\varepsilon = v^2/2 - \\mu/r$  [km$^2$/s$^2$]")
    axe.set_title("(b) Dissipazione di energia meccanica per attrito atmosferico",
                  fontsize=11.5)
    axe.set_xlim(0, t_med[-1] * 1.02)
    marg = 0.08 * (e_med[0] - e_med[-1])
    axe.set_ylim(e_med[-1] - marg, e_med[0] + marg)
    axe.grid(True, ls=":", alpha=0.6)

    de = e_med[-1] - e_med[0]
    axe.text(t_med[-1] * 0.02, e_med[0] + 0.3 * marg,
             f"energia all'inizio del rientro: {e_med[0]:.2f} km$^2$/s$^2$",
             color=C_INK2, fontsize=9.5, va="bottom")
    axe.annotate(f"$\\Delta\\varepsilon$ = {de:.2f} km$^2$/s$^2$   "
                 f"({100*abs(de/e_med[0]):.1f} % dell'energia iniziale)\n"
                 f"semiasse maggiore medio: da {a_med[0]:.0f} km "
                 f"a {a_med[-1]:.0f} km di quota",
                 xy=(t_med[-1], e_med[-1]),
                 xytext=(t_med[-1] * 0.46, e_med[0] - 0.62 * (e_med[0] - e_med[-1])),
                 color=C_CHASER, fontweight="bold", fontsize=10, ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_CHASER, lw=1.3))

    fig.suptitle("Fase 4 - Rientro passivo del sistema Chaser + Starlink "
                 "sotto l'effetto del drag atmosferico",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    salva(fig, "07_evoluzione_rientro")


# ===========================================================================
# FIGURA 8 - Analisi parametrica del rientro
# ===========================================================================
def figura_8_sensibilita_bc():
    print("  ...propagazioni parametriche su B_C (9 casi)")
    bc = np.sort(np.append([10, 20, 35, 70, 100, 130, 165, 200], P3.B_c))
    tempi = np.array([tempo_decadimento(H_TARGET, H_DEORBIT, b) for b in bc])
    t_missione = float(tempi[np.argmin(np.abs(bc - P3.B_c))])   # caso di missione

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 6.4))

    a1.plot(bc, tempi, "-o", color=C_CHASER, lw=2.4, ms=7, mec="white", mew=1.2,
            label="Propagazione numerica (DOP853 + US-76)")
    a1.plot(P3.B_c, t_missione, "*", ms=21, color=C_LIMIT, mec="white", mew=1.3,
            zorder=6, ls="none", label="Configurazione di missione")
    a1.annotate(f"$B_C$ = {P3.B_c:.1f} kg/m$^2$\n"
                f"rientro in {t_missione:.0f} giorni",
                xy=(P3.B_c, t_missione), xytext=(95, 14),
                color=C_LIMIT, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.3))
    a1.set_xlabel("Coefficiente balistico $B_C = m\\,/\\,(C_D A)$ [kg/m$^2$]")
    a1.set_ylabel("Tempo di rientro [giorni]")
    a1.set_title(f"(a) Effetto del coefficiente balistico\n"
                 f"orbita iniziale {H_TARGET:.0f} x {H_DEORBIT:.0f} km")
    a1.legend(loc="upper left")
    a1.grid(True, ls=":", alpha=0.6)

    # --- trade-off sulla quota del perigeo di rientro ---------------------
    print("  ...propagazioni parametriche sul perigeo (8 casi)")
    hp = np.array([120, 150, 180, 200, 230, 260, 290, 320], dtype=float)
    t_hp = np.array([tempo_decadimento(H_TARGET, q, P3.B_c) for q in hp])
    dv_hp = np.abs(np.sqrt(MU / P2.r_target) -
                   np.sqrt(2 * MU / P2.r_target -
                           MU / ((P2.r_target + RE + hp) / 2))) * 1000.0
    mp_hp = tsiolkovsky(P3.total_mass, dv_hp)

    a2.plot(hp, t_hp, "-o", color=C_TARGET, lw=2.4, ms=7, mec="white", mew=1.2)
    for q, tt, mm in zip(hp, t_hp, mp_hp):
        a2.annotate(f"{mm:.1f} kg", xy=(q, tt), xytext=(9, -4),
                    textcoords="offset points", ha="left", fontsize=9,
                    color=C_INK2)
    a2.plot(H_DEORBIT, float(np.interp(H_DEORBIT, hp, t_hp)), "*", ms=21,
            color=C_LIMIT, mec="white", mew=1.3, zorder=6)
    a2.annotate(f"scelta di progetto\nperigeo {H_DEORBIT:.0f} km\n"
                f"{P3.m_prop:.1f} kg $\\to$ {t_missione:.0f} giorni",
                xy=(H_DEORBIT, float(np.interp(H_DEORBIT, hp, t_hp))),
                xytext=(126, 95), color=C_LIMIT, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.3))
    a2.set_xlim(112, 352)
    a2.set_yscale("log")
    a2.set_xlabel("Quota del perigeo dopo il burn di deorbiting [km]")
    a2.set_ylabel("Tempo di rientro [giorni]")
    a2.set_title("(b) Trade-off: quanto conviene abbassare il perigeo")
    a2.text(0.985, 0.035, "le etichette indicano il propellente richiesto",
            transform=a2.transAxes, fontsize=9, color=C_INK2, ha="right")
    a2.grid(True, which="both", ls=":", alpha=0.6)

    fig.suptitle("Analisi parametrica del rientro: coefficiente balistico e "
                 "quota del perigeo", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    salva(fig, "08_sensibilita_rientro")


# ===========================================================================
# FIGURA 9 - Validazione del modello Python contro FreeFlyer
# ===========================================================================
ND = "n.d."


def _fmt(x, fmt):
    return ND if x is None else format(x, fmt)


def _diff(py, ff, fmt, perc=False):
    """Differenza FreeFlyer - Python, con eventuale scarto percentuale."""
    if py is None or ff is None:
        return ND
    d = ff - py
    d = 0.0 if round(d, int(fmt[1:-1])) == 0 else d      # niente "-0.000"
    if not perc:
        return f"{d:+{fmt}}"
    p = 100 * d / py
    p = 0.0 if round(p, 3) == 0 else p
    return f"{d:+{fmt}}   ({p:+.3f} %)"


def _tabella(ax, titolo, sottotitolo, righe):
    """Tabella Grandezza | Python | FreeFlyer | Differenza su un asse vuoto."""
    ax.axis("off")
    tab = ax.table(cellText=righe,
                   colLabels=["Grandezza", "Modello Python", "FreeFlyer",
                              "Differenza (FreeFlyer $-$ Python)"],
                   colWidths=[0.40, 0.16, 0.16, 0.28],
                   cellLoc="center", loc="upper center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.6)
    for (i, j), cella in tab.get_celld().items():
        cella.set_edgecolor(C_GRID)
        cella.set_linewidth(0.8)
        testo = cella.get_text()
        if j == 0:                     # prima colonna allineata a sinistra
            cella._loc = "left"
            cella.PAD = 0.03
            testo.set_horizontalalignment("left")
        if i == 0:
            cella.set_facecolor("#f1f0eb")
            testo.set_fontweight("bold")
            testo.set_color(C_INK)
        else:
            cella.set_facecolor("white" if i % 2 else "#fafaf7")
            testo.set_color(C_MUTED if testo.get_text() == ND else C_INK)
    ax.set_title(titolo, loc="left", pad=24)
    ax.text(0.0, 1.012, sottotitolo, transform=ax.transAxes, fontsize=9.5,
            color=C_INK2, va="bottom")


def figura_9_validazione():
    """Confronto Python / FreeFlyer in due tabelle.

    (a) I Delta-V degli oggetti ImpulsiveBurn di FreeFlyer SONO i valori calcolati
        da Python (inseriti a mano), con la stessa I_sp e lo stesso g: la
        concordanza verifica la trascrizione e il calcolo del propellente, non il
        modello.
    (b) Grandezze che FreeFlyer ottiene propagando da solo la traiettoria: questa
        e' la validazione vera e propria.
    """
    b1, b2, b3 = (FF_BURNS[n] for n in (1, 2, 3))
    m1 = P2.mass_chaser * np.exp(-DV1 / (ISP * G0))

    dati_a = [("$\\Delta V_1$ [m/s]", DV1, b1 and b1["dv"], ".3f"),
              ("$\\Delta V_2$ [m/s]", DV2, b2 and b2["dv"], ".3f"),
              ("$\\Delta V_3$ [m/s]", DV3, b3 and b3["dv"], ".3f"),
              ("Propellente $\\Delta V_1$ [kg]", P2.mass_chaser - m1,
               b1 and b1["m_prop"], ".3f"),
              ("Propellente $\\Delta V_2$ [kg]", m1 - P2.final_mass_chaser,
               b2 and b2["m_prop"], ".3f"),
              ("Propellente $\\Delta V_3$ [kg]", P3.m_prop, b3 and b3["m_prop"], ".3f"),
              ("Massa Chaser + Starlink prima del $\\Delta V_3$ [kg]", P3.total_mass,
               b3 and b3["m0"], ".3f")]
    righe_a = [[lab, _fmt(py, fmt), _fmt(ff, fmt), _diff(py, ff, fmt, perc=True)]
               for lab, py, ff, fmt in dati_a]

    t1 = (b1["epoca"] - FF_EPOCA0).total_seconds() if b1 and b1["epoca"] else None
    tof = (b2["epoca"] - b1["epoca"]).total_seconds() if b1 and b2 else None
    h2 = np.linalg.norm(b2["pre"][:3]) - RE if b2 else None
    apo2, peri2 = apsidi(b2["post"]) if b2 else (None, None)
    apo3, peri3 = apsidi(b3["post"]) if b3 else (None, None)
    rng = FF_RANGE[1] if FF_RANGE else None
    t_re_ff = ((FF_RIENTRO[0] - b3["epoca"]).total_seconds() / 86400.0
               if FF_RIENTRO and FF_RIENTRO[0] and b3 else None)
    t_re_py = (P4.solution.t_events[0][0] / 86400.0
               if P4.solution.t_events[0].size else None)

    dati_b = [("Istante del $\\Delta V_1$ dall'inizio [s]", P2.wt, t1, ".2f", False),
              ("Durata del trasferimento $\\Delta V_1 \\to \\Delta V_2$ [s]", P2.TOF, tof,
               ".2f", False),
              ("Quota al $\\Delta V_2$ [km]", H_TARGET, h2, ".3f", False),
              ("Distanza Chaser $-$ Starlink al $\\Delta V_2$ [km]", 0.0, rng, ".3f", False),
              ("Apogeo dopo il $\\Delta V_2$ [km]", H_TARGET, apo2, ".2f", False),
              ("Perigeo dopo il $\\Delta V_2$ [km]", H_TARGET, peri2, ".2f", False),
              ("Apogeo dopo il $\\Delta V_3$ [km]", H_TARGET, apo3, ".2f", False),
              ("Perigeo dopo il $\\Delta V_3$ [km]", H_DEORBIT, peri3, ".2f", False),
              (f"Tempo di rientro a {P4.h_reentry:.0f} km [giorni]", t_re_py, t_re_ff,
               ".2f", True)]
    righe_b = [[lab, _fmt(py, fmt), _fmt(ff, fmt), _diff(py, ff, fmt, perc)]
               for lab, py, ff, fmt, perc in dati_b]

    # altezza della figura proporzionale al numero di righe (~0.27 pollici a riga)
    alt = 0.27 * (len(righe_a) + len(righe_b) + 2) + 2.4
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, alt),
                                 gridspec_kw={"height_ratios": [len(righe_a) + 2.5,
                                                                len(righe_b) + 2.5]})
    _tabella(a1, "(a) Verifica di coerenza dei dati di input",
             "I $\\Delta V$ impostati in FreeFlyer sono quelli calcolati in Python "
             f"(arrotondati a 1 mm/s), con $I_{{sp}}$ = {ISP:.0f} s e g = {G0} m/s$^2$: "
             "la concordanza conferma la trascrizione, non il modello.",
             righe_a)
    _tabella(a2, "(b) Validazione: grandezze calcolate autonomamente da FreeFlyer",
             f"FreeFlyer: {FF_CONFIG}.   Python: Terra puntiforme (senza J2), "
             "drag US-76, integratore DOP853.",
             righe_b)
    if ND in (x for riga in righe_b for x in riga):
        fig.text(0.5, 0.01,
                 "n.d. = dato non ancora esportato da FreeFlyer (servono i report "
                 "\"report_range_rendezvous.txt\" e \"report_rientro.txt\" nella "
                 "cartella freeflyer)",
                 ha="center", fontsize=9, color=C_INK2)
    fig.suptitle("Confronto tra il modello Python e la simulazione FreeFlyer",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=3.0)
    salva(fig, "09_validazione_freeflyer")


# ===========================================================================
# FIGURA 10 - Evoluzione della massa lungo la missione
# ===========================================================================
def figura_10_bilancio_massa():
    m1 = P2.mass_chaser * np.exp(-DV1 / (ISP * G0))      # dopo Delta-V1
    m2 = m1 * np.exp(-DV2 / (ISP * G0))                  # dopo Delta-V2
    m_sys_cattura = m2 + P2.starlink_mass                # sistema agganciato
    m_sys_dopo_dv3 = m_sys_cattura - P3.m_prop           # sistema dopo Delta-V3
    m_chaser_fine = m_sys_dopo_dv3 - P2.starlink_mass    # solo chaser

    eventi = ["Lancio\n(orbita 400 km)", "dopo $\\Delta V_1$",
              "dopo $\\Delta V_2$\n(orbita 550 km)", "Cattura\ndel detrito",
              "dopo $\\Delta V_3$\n(deorbiting)", "Rientro\n(100 km)"]
    # prima della cattura il "sistema" coincide con il solo chaser
    masse_chaser = np.array([P2.mass_chaser, m1, m2, m2,
                             m_chaser_fine, m_chaser_fine])
    masse_totali = np.array([P2.mass_chaser, m1, m2, m_sys_cattura,
                             m_sys_dopo_dv3, m_sys_dopo_dv3])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(eventi))

    a1.step(x, masse_totali, where="post", color=C_TRANSFER, lw=2.8,
            label="Massa del sistema (Chaser + detrito)")
    a1.step(x, masse_chaser, where="post", color=C_CHASER, lw=2.8,
            label="Massa del solo Chaser")
    a1.plot(x, masse_totali, "o", ms=8, color=C_TRANSFER, mec="white", mew=1.3, zorder=6)
    a1.plot(x, masse_chaser, "o", ms=8, color=C_CHASER, mec="white", mew=1.3, zorder=6)
    for i, (mc, mt) in enumerate(zip(masse_chaser, masse_totali)):
        a1.annotate(f"{mt:.1f}", (x[i], mt), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=9.5,
                    color=C_TRANSFER, fontweight="bold")
        if abs(mt - mc) > 1:
            a1.annotate(f"{mc:.1f}", (x[i], mc), textcoords="offset points",
                        xytext=(0, -20), ha="center", fontsize=9.5,
                        color=C_CHASER, fontweight="bold")
    a1.annotate(f"+{P2.starlink_mass:.0f} kg\n(Starlink)",
                xy=(3, (m2 + m_sys_cattura) / 2), xytext=(2.25, 560),
                color=C_INK2, fontsize=10, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3))
    a1.set_xticks(x)
    a1.set_xticklabels(eventi, fontsize=9)
    a1.set_ylim(330, 740)
    a1.set_ylabel("Massa [kg]")
    a1.set_title("(a) Evoluzione della massa lungo la sequenza di missione")
    a1.legend(loc="upper left")
    a1.grid(True, axis="y", ls=":", alpha=0.6)

    # --- ripartizione del propellente ------------------------------------
    mp1, mp2 = P2.mass_chaser - m1, m1 - m2
    voci = ["$\\Delta V_1$  phasing", "$\\Delta V_2$  phasing",
            "$\\Delta V_3$  deorbiting"]
    valori = [mp1, mp2, P3.m_prop]
    for b, v, col in zip(a2.barh(voci, valori, color=[C_CHASER, C_TRANSFER, C_TARGET],
                                 height=0.55, zorder=3), valori,
                         (C_CHASER, C_TRANSFER, C_TARGET)):
        a2.text(v + 0.4, b.get_y() + b.get_height() / 2,
                f"{v:.2f} kg   ({100*v/sum(valori):.0f} %)", va="center",
                fontsize=10.5, color=C_INK, fontweight="bold")
    a2.set_xlim(0, max(valori) * 1.5)
    a2.set_xlabel("Propellente [kg]")
    a2.set_title(f"(b) Ripartizione del propellente\ntotale "
                 f"{P3.total_propellant_burned:.2f} kg su "
                 f"{P2.mass_chaser:.0f} kg iniziali "
                 f"({100*P3.total_propellant_burned/P2.mass_chaser:.1f} %)")
    a2.grid(True, axis="x", ls=":", alpha=0.6)
    a2.set_axisbelow(True)
    a2.invert_yaxis()

    fig.suptitle("Bilancio di massa della missione di Active Debris Removal",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    salva(fig, "10_bilancio_massa_missione")


# ===========================================================================
# Esecuzione
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("VERIFICA DI COERENZA DELLA GEOMETRIA DI PHASING")
    print("=" * 72)
    ang_t = np.mod(THETA_T0 + P2.n_T * T_RENDEZVOUS, 2 * np.pi)
    print(f"Angolo polare del Delta-V1:            {np.degrees(THETA_BURN):8.3f} deg")
    print(f"Angolo polare del Chaser al Delta-V2:  {np.degrees(THETA_DV2):8.3f} deg")
    print(f"Angolo polare del Target al Delta-V2:  {np.degrees(ang_t):8.3f} deg")
    print(f"Errore di rendez-vous:                 "
          f"{np.degrees(abs(THETA_DV2-ang_t)):8.2e} deg")
    print(f"\nFreeFlyer (dal MissionPlan): {FF_CONFIG}")
    for n in (1, 2, 3):
        b = FF_BURNS[n]
        if b:
            print(f"FreeFlyer Burn{n}: {b['epoca']}  dV = {b['dv']:7.3f} m/s, "
                  f"propellente = {b['m_prop']:6.3f} kg  "
                  f"({trova_file_freeflyer(f'Burn{n}.txt')})")
    if FF_RANGE:
        print(f"FreeFlyer distanza al Delta-V2: {FF_RANGE[1]:.3f} km")
    if FF_RIENTRO and FF_RIENTRO[0] and FF_BURNS[3]:
        print(f"FreeFlyer rientro: {FF_RIENTRO[0]} "
              f"({(FF_RIENTRO[0]-FF_BURNS[3]['epoca']).total_seconds()/86400:.2f} "
              f"giorni dopo il Delta-V3)")

    print("\n" + "=" * 72)
    print("GENERAZIONE DEI GRAFICI")
    print("=" * 72)
    _t = time.time()
    figura_1_phasing_3d()
    figura_2_geometria_phasing()
    figura_3_range()
    figura_4_massa_vs_quota()
    figura_5_densita()
    figura_6_decadimento()
    figura_7_evoluzione_orbita()
    figura_8_sensibilita_bc()
    figura_9_validazione()
    figura_10_bilancio_massa()
    print(f"\n10 figure generate in {time.time()-_t:.1f} s")
    print(f"Cartella di output: {OUTDIR}")

    plt.show()

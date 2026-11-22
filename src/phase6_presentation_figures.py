"""
phase6_presentation_figures.py
==============================
Figures for the thesis presentation (11-18), written to ../figures. They use the
same model as phases 2-4: masses, areas, C_D, US-76 atmosphere and Earth rotation
are all imported from the phase modules.

  11  Decay from the 550 x 200 km ellipse: target alone vs chaser + target
  12  Drag sail vs propulsive deorbit (target at 550 km), theoretical comparison
  13  Natural orbital lifetime as a function of altitude (Kessler context)
  14-18  Slide-sized versions of figures 02, 01-02, 10, 05 and 08
         (fonts >= 11 pt, Italian decimal comma)

Natural decay from circular orbits (figures 12-13) uses the orbit-averaged rate
    da/dt = -rho(h) * sqrt(mu*a) * (1 - w_E*r/v)^2 / B_C
which makes the lifetime proportional to B_C at equal density; the formula is
checked against the DOP853 propagation of phase 4 (printed to the terminal).

Figure labels are in Italian, as in the thesis.
Run:  python phase6_presentation_figures.py   (~1 minute)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUTDIR = os.path.join(HERE, "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)

import phase2_phasing as P2   # noqa: E402
import phase3_deorbit as P3   # noqa: E402
import phase4_decay as P4     # noqa: E402

MU = P4.mu
RE = P4.R_earth
W_E = P4.w_earth
H_REENTRY = P4.h_reentry
H_TARGET = P3.r_target - P3.r_earth
H_PERIGEE = P3.r_deorbit - P3.r_earth
H_CHASER = P2.r_chaser - P2.r_earth
ISP, G0 = P2.Isp, P2.g
DV1 = abs(P2.delta_v1_chaser) * 1000.0   # m/s
DV2 = abs(P2.delta_v2_chaser) * 1000.0   # m/s

M_TARGET = P3.starlink_mass
CDA_TARGET = P3.Cd_target * P3.frontal_area_starlink
BC_TARGET = M_TARGET / CDA_TARGET          # solo target
BC_SYSTEM = P3.B_c                         # Chaser + target dopo il burn

# --- Stile (stessa palette di phase5_figures.py) ---------------------------
C_CHASER, C_TARGET, C_TRANSFER, C_LIMIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
C_INK, C_INK2, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": C_INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.7,
    "xtick.color": C_INK2, "ytick.color": C_INK2,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": C_GRID, "legend.fontsize": 9.5,
    "font.size": 10.5, "lines.linewidth": 2.0,
})
# Figure 14-18: dimensioni in pollici pari al riquadro della slide, quindi i
# caratteri appaiono in proiezione con la dimensione indicata qui.
SLIDE_RC = {"font.size": 12, "axes.labelsize": 12.5, "axes.titlesize": 12.5,
            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11}


def salva(fig, nome):
    path = os.path.join(OUTDIR, nome + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [ok] {nome}.png")


def it(x, nd=1):
    """Numero con la virgola decimale, come nel testo delle slide."""
    return f"{x:.{nd}f}".replace(".", ",")


_VIRGOLA = FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ",").replace("-", "−"))


def virgola(ax):
    """Separatore decimale a virgola sulle etichette degli assi lineari."""
    if ax.get_xscale() == "linear":
        ax.xaxis.set_major_formatter(_VIRGOLA)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_formatter(_VIRGOLA)


# --- Propagazione numerica (stesse equazioni di phase4_decay.kepler_2d) ----------
def rhs(t, s, bc):
    x, y, vx, vy = s
    r = np.hypot(x, y)
    rho = P4.density(r - RE)
    v_rel = (np.array([vx, vy]) - np.array([-W_E * y, W_E * x])) * 1000.0
    v_n = np.hypot(*v_rel)
    a_drag = -0.5 * rho * v_n * v_rel / bc / 1000.0
    return [vx, vy, -MU * x / r**3 + a_drag[0], -MU * y / r**3 + a_drag[1]]


def reentry(t, s, bc):
    return np.hypot(s[0], s[1]) - RE - H_REENTRY
reentry.terminal = True
reentry.direction = -1


def propaga(state0, bc, t_max):
    sol = solve_ivp(rhs, [0, t_max], state0, method="DOP853", rtol=1e-8, atol=1e-9,
                    t_eval=np.arange(0, t_max, 60.0), events=reentry, args=(bc,))
    h = np.hypot(sol.y[0], sol.y[1]) - RE
    t_re = sol.t_events[0][0] if sol.t_events[0].size else np.nan
    # rivoluzioni effettive (il periodo si accorcia durante il decadimento)
    n_rev = np.unwrap(np.arctan2(sol.y[1], sol.y[0]))[-1] / (2 * np.pi)
    return sol.t, h, t_re, n_rev


def tempo_decadimento(h_apo, h_peri, bc, t_max_giorni=1500):
    """Tempo di rientro [giorni] dall'apogeo dell'ellisse h_apo x h_peri.

    Stesse tolleranze di phase5_figures.tempo_decadimento (figura 08), per
    ottenere gli stessi valori.
    """
    ra, rp = RE + h_apo, RE + h_peri
    a = (ra + rp) / 2.0
    s0 = [ra, 0.0, 0.0, np.sqrt(MU * (2 / ra - 1 / a))]
    sol = solve_ivp(rhs, [0, t_max_giorni * 86400], s0, method="DOP853",
                    rtol=1e-7, atol=1e-8, events=reentry, args=(bc,))
    return sol.t_events[0][0] / 86400.0 if sol.t_events[0].size else np.nan


def inviluppi(t, h):
    """Apogeo e perigeo di ogni rivoluzione (massimi e minimi locali di h)."""
    ip, _ = find_peaks(h, distance=40)
    iq, _ = find_peaks(-h, distance=40)
    return (t[ip], h[ip]), (t[iq], h[iq])


# --- Decadimento naturale mediato sull'orbita (orbita circolare) -----------
def tempo_decadimento_circolare(h0, bc):
    """Tempo [s] per scendere da h0 a H_REENTRY su orbita circolare."""
    def dadt(t, a):
        a_m = a[0] * 1000.0
        v = np.sqrt(MU * 1e9 / a_m)
        f = (1.0 - W_E * a[0] / (v / 1000.0))**2
        return [-P4.density(a[0] - RE) * np.sqrt(MU * 1e9 * a_m) * f / bc / 1000.0]

    def fine(t, a):
        return a[0] - RE - H_REENTRY
    fine.terminal = True
    sol = solve_ivp(dadt, [0, 200 * 365.25 * 86400], [RE + h0], events=fine,
                    rtol=1e-9, atol=1e-6, max_step=86400 * 5)
    return sol.t_events[0][0]


# --- Geometria del phasing (come in phase5_figures.py) ----------------------
# Il chaser parte con anomalia vera 0 (asse +X), il target con 45 gradi, come
# nella scena FreeFlyer. Gli angoli polari sono riportati in [0, 2*pi).
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


def anomalia_vera(t, a, e, mu=P2.mu):
    """Anomalia vera dopo un tempo t dal passaggio al perigeo."""
    n = np.sqrt(mu / a ** 3)
    E = kepler_E(n * np.asarray(t, dtype=float), e)
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


THETA_BURN = np.mod(P2.n_C * P2.wt, 2 * np.pi)                 # angolo polare del Delta-V1
THETA_T0 = np.radians(P2.delta_phi_0)                          # target a t = 0
THETA_T_BURN = np.mod(THETA_T0 + P2.n_T * P2.wt, 2 * np.pi)    # target al Delta-V1
THETA_DV2 = np.mod(THETA_BURN + np.pi, 2 * np.pi)              # angolo polare del Delta-V2
T_RENDEZVOUS = P2.wt + P2.TOF


def stato_chaser(t):
    """Raggio [km] e angolo polare del chaser durante attesa + trasferimento."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    r = np.empty_like(t)
    ang = np.empty_like(t)
    coast = t <= P2.wt
    r[coast] = P2.r_chaser
    ang[coast] = P2.n_C * t[coast]
    tr = ~coast
    nu = anomalia_vera(t[tr] - P2.wt, P2.a_phas, P2.e_phas)
    r[tr] = P2.a_phas * (1 - P2.e_phas ** 2) / (1 + P2.e_phas * np.cos(nu))
    ang[tr] = THETA_BURN + nu
    return r, ang


def stato_target(t):
    """Raggio [km] e angolo polare del target (orbita circolare a 550 km)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.full_like(t, P2.r_target), THETA_T0 + P2.n_T * t


# ===========================================================================
print("Verifica formula mediata vs propagazione numerica (circolare 250 km)...")
r0 = RE + 250.0
t_num = propaga([r0, 0, 0, np.sqrt(MU / r0)], BC_SYSTEM, 60 * 86400)[2]
t_avg = tempo_decadimento_circolare(250.0, BC_SYSTEM)
print(f"  numerico {t_num/86400:.2f} giorni - mediato {t_avg/86400:.2f} giorni "
      f"(scarto {100*(t_avg-t_num)/t_num:+.1f} %)")

# --- Figura 11: solo target vs Chaser + target ------------------------------
print("Propagazione del rientro dall'ellisse "
      f"{H_TARGET:.0f} x {H_PERIGEE:.0f} km...")
casi = [
    ("Chaser + target (catturato)", BC_SYSTEM, C_TRANSFER),
    ("solo target (rilasciato dopo il burn)", BC_TARGET, C_TARGET),
]
risultati = []
for nome, bc, col in casi:
    t, h, t_re, n_rev = propaga(P4.state_vector, bc, 3 * 365 * 86400)
    risultati.append((nome, bc, col, t, h, t_re))
    print(f"  {nome:40s} B_C = {bc:6.2f} kg/m^2  ->  rientro in {t_re/86400:6.2f} giorni "
          f"({n_rev:.0f} rivoluzioni)")

fig, ax = plt.subplots(figsize=(10, 5.4))
for nome, bc, col, t, h, t_re in risultati:
    (ta, ha), (tp, hp) = inviluppi(t, h)
    ax.plot(ta / 86400, ha, color=col, lw=2.4,
            label=f"{nome}: $B_C$ = {it(bc)} kg/m$^2$")
    ax.plot(tp / 86400, hp, color=col, lw=2.4, ls="--")
    ax.plot(t_re / 86400, H_REENTRY, "o", color=col, ms=9, mec="white", mew=1.5, zorder=5)
    ax.text(t_re / 86400 - 0.3, H_REENTRY - 22, f"rientro dopo {it(t_re/86400)} giorni",
            ha="right", va="center", color=col, fontweight="bold", fontsize=11)
ax.axhline(H_REENTRY, color=C_LIMIT, ls="--", lw=1.4, label=f"rientro convenzionale ({H_REENTRY} km)")
ax.plot([], [], color=C_MUTED, lw=2.4, label="apogeo (linea continua)")
ax.plot([], [], color=C_MUTED, lw=2.4, ls="--", label="perigeo (linea tratteggiata)")
ax.set_xlabel("Tempo dal burn di deorbiting [giorni]")
ax.set_ylabel("Quota [km]")
ax.set_ylim(50, 700)
ax.set_xlim(left=0)
ax.set_title(f"Decadimento dall'ellisse {H_TARGET:.0f} × {H_PERIGEE:.0f} km: "
             "solo target vs sistema Chaser + target")
ax.legend(loc="upper right")
virgola(ax)
salva(fig, "11_confronto_decadimento_target_sistema")
plt.close(fig)

# --- Figura 12: vela di drag vs deorbiting propulsivo -----------------------
print("Decadimento naturale del target da orbita circolare...")
t_nat_target = tempo_decadimento_circolare(H_TARGET, BC_TARGET)
# T proporzionale a B_C = m / (C_D A)  ->  T(A) = T_nat * A_target / A
K = t_nat_target / BC_TARGET
A_starlink = P3.frontal_area_starlink
cd = P3.Cd_target
anno = 365.25 * 86400
t_adr = risultati[0][5]

obiettivi = [
    (25 * anno, "25 anni (IADC)"),
    (5 * anno, "5 anni (ESA 2023 / FCC)"),
    (1 * anno, "1 anno"),
    (t_adr, f"{t_adr/86400:.0f} giorni (ADR, questo lavoro)"),
]
print(f"  target senza vela (A = {A_starlink:.0f} m^2): {t_nat_target/anno:.2f} anni")
righe = []
for t_obj, lab in obiettivi:
    a_tot = M_TARGET * K / (cd * t_obj)
    righe.append((t_obj, lab, a_tot))
    print(f"  rientro in {lab:45s}: area totale {a_tot:8.1f} m^2 -> vela {max(a_tot-A_starlink,0):8.1f} m^2")

# Figura compatta con font grandi: va inserita in mezza slide.
with plt.rc_context({"font.size": 12.5, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11.5}):
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    A = np.logspace(0, 3.2, 300)
    T = K * M_TARGET / (cd * A)
    ax.loglog(A, T / 86400, color=C_TARGET, lw=3, label="target da 550 km (orbita circolare)")
    ax.plot(A_starlink, t_nat_target / 86400, "o", color=C_INK2, ms=11, mec="white", mew=1.5, zorder=6)
    ax.annotate(f"senza vela (A = {A_starlink:.0f} m$^2$):\n{it(t_nat_target/anno)} anni",
                xy=(A_starlink, t_nat_target / 86400), xytext=(9, 3300),
                color=C_INK2, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_INK2, lw=1.3))
    for t_obj, lab, a_tot in righe:
        adr = "ADR" in lab
        col = C_TRANSFER if adr else C_CHASER
        ax.axhline(t_obj / 86400, color=col, ls=":", lw=1.4)
        if a_tot > 1.5 * A_starlink:   # serve davvero una vela
            ax.plot(a_tot, t_obj / 86400, "*" if adr else "s", color=col,
                    ms=18 if adr else 10, mec="white", mew=1.2, zorder=5)
            ax.annotate(f"A = {a_tot:.0f} m$^2$", xy=(a_tot, t_obj / 86400),
                        xytext=(9, 9), textcoords="offset points", color=col,
                        fontweight="bold", fontsize=13)
        elif a_tot <= A_starlink:
            lab += ": già soddisfatto"
        else:
            lab += f": vela di {it(a_tot - A_starlink)} m$^2$"
        alto = t_obj > 2 * 365.25 * 86400
        ax.text(10**3.17 if alto else 1.12, t_obj / 86400 * 1.15, lab, color=col, fontsize=11.5,
                ha="right" if alto else "left", fontweight="bold")
    ax.set_xlabel("Area di drag totale $A$ [m$^2$]")
    ax.set_ylabel("Tempo di rientro [giorni]")
    ax.set_xlim(1, 10**3.2)
    ax.set_ylim(4, 2.2e4)
    ax.legend(loc="lower left")
    salva(fig, "12_vela_drag_vs_propulsione")
    plt.close(fig)

# --- Figura 13: tempo di permanenza naturale in funzione della quota ---------
print("Tempo di permanenza naturale in funzione della quota...")
BC_COMPATTO = 100.0   # kg/m^2: oggetto piu' denso del caso di studio (es. stadio di razzo)
EVENTI = [(480, "Cosmos 1408 (2021)"), (790, "Iridium-Cosmos (2009)"), (865, "Fengyun-1C (2007)")]


def permanenza_circolare(bc, h_max=1000.0, dh=0.05):
    """Tempo [anni] per decadere da h a H_REENTRY su orbita circolare, per h in [H_REENTRY, h_max].

    Stessa formula mediata sull'orbita di tempo_decadimento_circolare(), integrata in
    quota invece che nel tempo: t(h0) = integrale da H_REENTRY a h0 di dh / |dh/dt|.
    """
    h = np.arange(H_REENTRY, h_max + dh, dh)
    a_m = (RE + h) * 1000.0
    v = np.sqrt(MU * 1e9 / a_m)
    f = (1.0 - W_E * a_m / v) ** 2
    rho = np.array([P4.density(x) for x in h])
    dtdh = 1000.0 / (rho * np.sqrt(MU * 1e9 * a_m) * f / bc)      # s/km
    t = np.concatenate([[0.0], np.cumsum(0.5 * (dtdh[1:] + dtdh[:-1]) * dh)])
    return h, t / anno


h13, t_sl = permanenza_circolare(BC_TARGET)
_, t_cp = permanenza_circolare(BC_COMPATTO)
t550 = np.interp(H_TARGET, h13, t_sl)
print(f"  verifica a {H_TARGET:.0f} km: {t550:.2f} anni (integrazione nel tempo: {t_nat_target/anno:.2f})")
for hq in (500, 550, 600, 650, 700, 790, 865, 1000):
    print(f"  {hq:5.0f} km: {np.interp(hq, h13, t_sl):9.1f} anni (B_C {BC_TARGET:.1f})"
          f"  {np.interp(hq, h13, t_cp):9.1f} anni (B_C {BC_COMPATTO:.0f})")

with plt.rc_context({"font.size": 12.5, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11}):
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    sel = h13 >= 350
    ax.semilogy(h13[sel], t_sl[sel], color=C_TARGET, lw=3,
                label=f"caso di studio, $B_C$ = {it(BC_TARGET)} kg/m$^2$")
    ax.semilogy(h13[sel], t_cp[sel], color=C_INK2, lw=3, ls="--",
                label=f"oggetto più compatto, $B_C$ = {BC_COMPATTO:.0f} kg/m$^2$")
    for anni, lab in ((25, "25 anni (IADC)"), (5, "5 anni (ESA 2023 / FCC)")):
        ax.axhline(anni, color=C_CHASER, ls=":", lw=1.4)
        ax.text(995, anni * 1.16, lab, color=C_CHASER, fontsize=11.5, fontweight="bold", ha="right")
    # quote dei principali eventi di frammentazione, sulla curva del caso di studio
    h_ev = [hq for hq, _ in EVENTI]
    ax.plot(h_ev, np.interp(h_ev, h13, t_sl), "^", color=C_LIMIT, ms=10, mec="white", mew=1.2,
            zorder=6, label="eventi di frammentazione")
    for hq, lab in EVENTI:
        anno_ev = lab[lab.index("(") + 1:lab.index(")")]
        ax.text(hq + 8, np.interp(hq, h13, t_sl) * 0.62, anno_ev, color=C_LIMIT, fontsize=11.5,
                fontweight="bold", ha="left", va="top")
    ax.plot(H_TARGET, t550, "*", color=C_TRANSFER, ms=19, mec="white", mew=1.2, zorder=6)
    ax.annotate(f"{H_TARGET:.0f} km: ≈ {t550:.0f} anni", xy=(H_TARGET, t550),
                xytext=(H_TARGET + 30, t550 / 7), color=C_TRANSFER, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_TRANSFER, lw=1.3))
    ax.set_xlabel("Quota dell'orbita circolare [km]")
    ax.set_ylabel("Tempo di permanenza in orbita [anni]")
    ax.set_xlim(350, 1000)
    ax.set_ylim(0.1, 2e4)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    salva(fig, "13_permanenza_vs_quota")
    plt.close(fig)

# ===========================================================================
# Figure 14-18: versioni per la proiezione
# ===========================================================================
wt_h, tr_h = P2.wt / 3600, T_RENDEZVOUS / 3600
lead = np.degrees(P2.phi)
attesa = f"{int(P2.wt)//3600} h {int(P2.wt % 3600)//60} min"

# --- Figura 14: geometria del phasing, asse radiale = quota (slide 6) -------
print("Figure per la proiezione...")
with plt.rc_context(SLIDE_RC):
    fig = plt.figure(figsize=(6.6, 4.3))
    ax = fig.add_axes([0.0, 0.06, 0.6, 0.88], projection="polar")
    th = np.linspace(0, 2 * np.pi, 721)
    ax.plot(th, np.full_like(th, H_CHASER), color=C_CHASER, ls="--", lw=1.6,
            label=f"orbita Chaser ({H_CHASER:.0f} km)")
    ax.plot(th, np.full_like(th, H_TARGET), color=C_TARGET, ls="--", lw=1.6,
            label=f"orbita target ({H_TARGET:.0f} km)")
    th_wait = np.linspace(0, THETA_BURN, 400)
    ax.plot(th_wait, np.full_like(th_wait, H_CHASER), color=C_CHASER, lw=3.4,
            label=f"attesa (ultimi {it(THETA_BURN / (2 * np.pi), 2)} giri)")
    nu = np.linspace(0, np.pi, 400)
    r_tr = P2.a_phas * (1 - P2.e_phas ** 2) / (1 + P2.e_phas * np.cos(nu))
    ax.plot(THETA_BURN + nu, r_tr - P2.r_earth, color=C_TRANSFER, lw=3.4,
            label="trasferimento di Hohmann")
    punti = ((0.0, H_CHASER, "o", C_CHASER, 10, "Chaser a t = 0"),
             (THETA_T0, H_TARGET, "o", C_TARGET, 10, "target a t = 0"),
             (THETA_BURN, H_CHASER, "^", C_LIMIT, 13, f"$\\Delta V_1$ dopo {attesa}"),
             (THETA_T_BURN, H_TARGET, "s", C_TARGET, 9, "target al $\\Delta V_1$"),
             (THETA_DV2, H_TARGET, "v", C_LIMIT, 13, "$\\Delta V_2$: rendez-vous"))
    for ang, r, mk, col, ms, lab in punti:
        ax.plot(ang, r, mk, ms=ms, color=col, mec="white", mew=1.4, zorder=6, ls="none", label=lab)
    r_arc = 605
    arc0 = np.linspace(0, THETA_T0, 60)
    ax.plot(arc0, np.full_like(arc0, r_arc), color=C_MUTED, lw=2.6,
            label="fase iniziale $\\Delta\\varphi_0$ = 45°")
    arc1 = np.linspace(THETA_BURN, THETA_T_BURN, 30)
    ax.plot(arc1, np.full_like(arc1, r_arc), color=C_LIMIT, lw=3.4,
            label=f"lead angle $\\varphi$ = {it(lead)}°")
    ax.set_rlim(250, 640)
    ax.set_rticks([H_CHASER, H_TARGET])
    ax.set_yticklabels([f"{H_CHASER:.0f} km", f"{H_TARGET:.0f} km"], fontsize=10.5, color=C_INK2)
    for lbl in ax.get_yticklabels():
        lbl.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
    ax.set_rlabel_position(112)
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"], fontsize=10.5)
    ax.grid(True, ls=":", alpha=0.6)
    h_leg, l_leg = ax.get_legend_handles_labels()
    fig.legend(h_leg, l_leg, loc="center left", bbox_to_anchor=(0.61, 0.55), frameon=False,
               fontsize=11, handlelength=1.8, labelspacing=0.5)
    fig.text(0.625, 0.05, "asse radiale: quota [km]", ha="left", fontsize=10.5,
             style="italic", color=C_INK2)
    salva(fig, "14_slide_geometria_phasing")
    plt.close(fig)

# --- Figura 15: quota del Chaser e angolo di fase nel tempo (slide 7) -------
tt = np.linspace(0, T_RENDEZVOUS, 6000)
r_c, a_c = stato_chaser(tt)
_, a_t = stato_target(tt)
dphi = np.degrees(np.mod(a_t - a_c + np.pi, 2 * np.pi) - np.pi)
with plt.rc_context(SLIDE_RC):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.35), sharex=True,
                                 gridspec_kw={"hspace": 0.1, "height_ratios": [1, 1.15]})
    for ax in (a1, a2):
        ax.axvspan(0, wt_h, color=C_CHASER, alpha=0.07, lw=0)
        ax.axvspan(wt_h, tr_h, color=C_TRANSFER, alpha=0.16, lw=0)
        ax.axvline(wt_h, color=C_LIMIT, ls="-.", lw=1.3)
        ax.axvline(tr_h, color=C_LIMIT, ls="-.", lw=1.3)
        ax.grid(True, ls=":", alpha=0.6)
    a1.plot(tt / 3600, r_c - P2.r_earth, color=C_INK, lw=2.6)
    a1.axhline(H_CHASER, color=C_CHASER, ls="--", lw=1.3)
    a1.axhline(H_TARGET, color=C_TARGET, ls="--", lw=1.3)
    a1.text(0.12, H_CHASER + 9, f"{H_CHASER:.0f} km", color=C_CHASER, fontweight="bold")
    a1.text(0.12, H_TARGET - 11, f"{H_TARGET:.0f} km", color=C_TARGET, fontweight="bold", va="top")
    a1.text(wt_h / 2, 470, f"attesa a {H_CHASER:.0f} km\n{attesa} ({it(P2.wt / P2.T_chaser)} giri)",
            color=C_CHASER, ha="center", va="center", fontweight="bold")
    a1.text(tr_h + 0.08, 470, f"trasferimento\n{P2.TOF / 60:.0f} min", color=C_TRANSFER,
            ha="left", va="center", fontweight="bold")
    for tv, lab in ((wt_h, "$\\Delta V_1$"), (tr_h, "$\\Delta V_2$")):
        a1.text(tv, 582, lab, color=C_LIMIT, fontweight="bold", ha="center", va="bottom")
    a1.set_ylim(380, 580)
    a1.set_ylabel("Quota del\nChaser [km]")
    a2.plot(tt / 3600, dphi, color=C_TRANSFER, lw=2.8)
    a2.axhline(lead, color=C_MUTED, ls=":", lw=1.6)
    a2.axhline(0, color=C_LIMIT, ls="--", lw=1.2)
    a2.annotate("", xy=(0.25, lead), xytext=(0.25, P2.delta_phi_0),
                arrowprops=dict(arrowstyle="<->", color=C_CHASER, lw=1.8))
    a2.text(0.45, 11, f"{it(P2.delta_phi_0 - lead)}° recuperati nell'attesa", color=C_CHASER,
            fontweight="bold", ha="left", va="center")
    a2.text(tr_h + 0.08, 9, f"{it(lead)}° chiusi\nnel trasferimento", color=C_TRANSFER,
            fontweight="bold", ha="left", va="center")
    a2.text(0.45, lead + 1.2, f"lead angle {it(lead)}°", color=C_INK2, fontsize=10.5, va="bottom")
    a2.set_ylim(-4, 50)
    a2.set_ylabel("Angolo di fase\ntarget − Chaser [°]")
    a2.set_xlabel("Tempo dall'inizio [h]")
    a2.set_xlim(0, 7.9)
    virgola(a1)
    virgola(a2)
    salva(fig, "15_slide_quota_e_fase")
    plt.close(fig)

# --- Figura 16: evoluzione della massa (slide 8) ----------------------------
m1 = P2.mass_chaser * np.exp(-DV1 / (ISP * G0))
m2 = m1 * np.exp(-DV2 / (ISP * G0))
m_sys = m2 + M_TARGET
m_sys3 = m_sys - P3.m_prop
m_ch_fine = m_sys3 - M_TARGET
eventi = ["Lancio\n400 km", "dopo\n$\\Delta V_1$", "dopo $\\Delta V_2$\n550 km", "Cattura",
          "dopo\n$\\Delta V_3$", "Rientro\n100 km"]
masse_ch = np.array([P2.mass_chaser, m1, m2, m2, m_ch_fine, m_ch_fine])
masse_tot = np.array([P2.mass_chaser, m1, m2, m_sys, m_sys3, m_sys3])
with plt.rc_context(SLIDE_RC):
    fig, ax = plt.subplots(figsize=(6.3, 3.95))
    x = np.arange(len(eventi))
    ax.step(x, masse_tot, where="post", color=C_TRANSFER, lw=3, label="Chaser + target")
    ax.step(x, masse_ch, where="post", color=C_CHASER, lw=3, label="solo Chaser")
    ax.plot(x[3:], masse_tot[3:], "o", ms=8, color=C_TRANSFER, mec="white", mew=1.3, zorder=6)
    ax.plot(x, masse_ch, "o", ms=8, color=C_CHASER, mec="white", mew=1.3, zorder=6)
    for i in range(3):
        ax.annotate(it(masse_ch[i]), (x[i], masse_ch[i]), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=C_CHASER, fontweight="bold")
    ax.annotate(it(masse_tot[3]), (x[3], masse_tot[3]), textcoords="offset points",
                xytext=(0, 10), ha="center", color=C_TRANSFER, fontweight="bold")
    for i in (4, 5):
        ax.annotate(it(masse_tot[i]), (x[i], masse_tot[i]), textcoords="offset points",
                    xytext=(0, -20), ha="center", color=C_TRANSFER, fontweight="bold")
    for i in (3, 4, 5):
        ax.annotate(it(masse_ch[i]), (x[i], masse_ch[i]), textcoords="offset points",
                    xytext=(0, -20), ha="center", color=C_CHASER, fontweight="bold")
    ax.annotate(f"+{M_TARGET:.0f} kg\n(target)", xy=(3, (m2 + m_sys) / 2), xytext=(2.15, 560),
                color=C_INK2, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3))
    ax.set_xticks(x)
    ax.set_xticklabels(eventi, fontsize=11)
    ax.set_xlim(-0.4, 5.4)
    ax.set_ylim(320, 745)
    ax.set_ylabel("Massa [kg]")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    ax.grid(False, axis="x")
    ax.yaxis.set_major_formatter(_VIRGOLA)
    salva(fig, "16_slide_massa_missione")
    plt.close(fig)

# --- Figura 17: profilo di densita' US-76 (slide 10) -------------------------
hd = np.linspace(80, 700, 1500)
rho_d = np.array([P4.density(q) for q in hd])
q_miss = [H_REENTRY, H_PERIGEE, H_CHASER, H_TARGET]
with plt.rc_context(SLIDE_RC):
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    ax.semilogy(hd, rho_d, color=C_CHASER, lw=2.8,
                label="US Standard Atmosphere 1976 (esponenziale a tratti)")
    ax.plot(q_miss, [P4.density(q) for q in q_miss], "s", ms=9, color=C_LIMIT, mec="white",
            mew=1.2, zorder=6, ls="none", label="quote della missione (valori nei riquadri)")
    ax.set_xlim(80, 700)
    ax.set_xlabel("Quota [km]")
    ax.set_ylabel("Densità $\\rho$ [kg/m$^3$]")
    ax.legend(loc="upper right")
    ax.grid(True, which="major", ls=":", alpha=0.6)
    virgola(ax)
    salva(fig, "17_slide_densita")
    plt.close(fig)

# --- Figura 18: sensibilita' del rientro (slide 12) --------------------------
print("  ...propagazioni parametriche su B_C e sul perigeo (17 casi)")
bc_list = np.sort(np.append([10, 20, 35, 70, 100, 130, 165, 200], BC_SYSTEM))
t_bc = np.array([tempo_decadimento(H_TARGET, H_PERIGEE, b) for b in bc_list])
t_miss = float(t_bc[np.argmin(np.abs(bc_list - BC_SYSTEM))])
hp_list = np.array([120, 150, 180, 200, 230, 260, 290, 320], dtype=float)
t_hp = np.array([tempo_decadimento(H_TARGET, q, BC_SYSTEM) for q in hp_list])
dv_hp = np.abs(np.sqrt(P2.mu / P2.r_target) -
               np.sqrt(2 * P2.mu / P2.r_target -
                       P2.mu / ((P2.r_target + P2.r_earth + hp_list) / 2))) * 1000.0
mp_hp = P3.total_mass * (1.0 - np.exp(-dv_hp / (ISP * G0)))
for q, tq, mq in zip(hp_list, t_hp, mp_hp):
    print(f"     perigeo {q:5.0f} km: {tq:8.2f} giorni, {mq:5.2f} kg")
with plt.rc_context(SLIDE_RC):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.2), gridspec_kw={"wspace": 0.22})
    a1.plot(bc_list, t_bc, "-o", color=C_CHASER, lw=2.8, ms=8, mec="white", mew=1.2)
    a1.plot(BC_SYSTEM, t_miss, "*", ms=22, color=C_LIMIT, mec="white", mew=1.3, zorder=6)
    a1.annotate(f"missione: $B_C$ = {it(BC_SYSTEM)} kg/m$^2$\nrientro in {t_miss:.0f} giorni",
                xy=(BC_SYSTEM, t_miss), xytext=(88, 12), color=C_LIMIT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.4))
    a1.set_xlim(0, 210)
    a1.set_ylim(0, 100)
    a1.set_xlabel("Coefficiente balistico $B_C = m\\,/\\,(C_D A)$ [kg/m$^2$]")
    a1.set_ylabel("Tempo di rientro [giorni]")
    a1.set_title(f"(a) Effetto di $B_C$ (ellisse {H_TARGET:.0f} × {H_PERIGEE:.0f} km)", loc="left")
    a1.grid(True, ls=":", alpha=0.6)
    virgola(a1)
    a2.plot(hp_list, t_hp, "-o", color=C_TARGET, lw=2.8, ms=8, mec="white", mew=1.2)
    for q, tq, mq in zip(hp_list, t_hp, mp_hp):
        if abs(q - H_PERIGEE) > 1:
            a2.annotate(f"{it(mq)} kg", xy=(q, tq), xytext=(10, -5), textcoords="offset points",
                        ha="left", fontsize=11, color=C_INK2)
    t200 = float(np.interp(H_PERIGEE, hp_list, t_hp))
    a2.plot(H_PERIGEE, t200, "*", ms=22, color=C_LIMIT, mec="white", mew=1.3, zorder=6)
    a2.annotate(f"scelta: perigeo {H_PERIGEE:.0f} km\n{it(P3.m_prop)} kg → {t_miss:.0f} giorni",
                xy=(H_PERIGEE, t200), xytext=(118, 120), color=C_LIMIT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_LIMIT, lw=1.4))
    a2.set_yscale("log")
    a2.set_xlim(110, 345)
    a2.set_ylim(0.3, 600)
    a2.set_xlabel("Quota del perigeo dopo il burn di deorbiting [km]")
    a2.set_ylabel("Tempo di rientro [giorni]")
    a2.set_title("(b) Quota del perigeo: rientro e propellente", loc="left")
    a2.text(0.985, 0.04, "etichette: propellente del burn di deorbiting", transform=a2.transAxes,
            fontsize=10.5, color=C_INK2, ha="right")
    a2.grid(True, which="both", ls=":", alpha=0.5)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
    virgola(a2)
    salva(fig, "18_slide_sensibilita_rientro")
    plt.close(fig)

print("\nRiepilogo:")
print(f"  B_C solo target        = {BC_TARGET:.2f} kg/m^2")
print(f"  B_C Chaser + target    = {BC_SYSTEM:.2f} kg/m^2")
print(f"  rientro naturale target da {H_TARGET:.0f} km circolare: {t_nat_target/anno:.2f} anni")
print(f"  missione, rientro (propagazione fig. 18): {t_miss:.2f} giorni")

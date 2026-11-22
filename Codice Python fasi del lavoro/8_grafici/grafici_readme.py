"""
grafici_readme.py
=================
English figures for the GitHub README (folder "Grafici e immagini/readme/").
The thesis figures (grafici_tesi.py, grafici_presentazione.py) stay in Italian.

No physical parameter is redefined here: everything comes from phasing.py,
deorbiting.py and rientro.py, and the solar-activity scenarios are read from
Risultati/rientro_scenari_solari.csv (written by 5_rientro/rientro_scenari_solari.py).

  readme_phasing.png          Hohmann phasing 400 km -> 550 km: geometry and altitude profile
  readme_decay.png            Re-entry of the chaser + target stack from the 550 x 200 km ellipse
  readme_solar_activity.png   Time to re-entry with ADR vs natural decay, for three solar-activity levels

Run:  python 8_grafici/grafici_readme.py   (~10 s)
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Comuni"))
import percorsi
import phasing as P2      # noqa: E402
import deorbiting as P3   # noqa: E402
import rientro as P4      # noqa: E402

OUTDIR = os.path.join(percorsi.DIR_IMMAGINI, "readme")
os.makedirs(OUTDIR, exist_ok=True)

# Same palette as the thesis figures (validated for colour-vision deficiency)
C_CHASER, C_TARGET, C_TRANSFER, C_LIMIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
C_INK, C_INK2, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": C_INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.7,
    "xtick.color": C_INK2, "ytick.color": C_INK2,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": C_GRID,
    "font.size": 10.5, "lines.linewidth": 2.0,
})

RE = P2.r_earth
H_CHASER = P2.r_chaser - RE
H_TARGET = P2.r_target - RE


def save(fig, name):
    path = os.path.join(OUTDIR, name + ".png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] readme/{name}.png")


# ---------------------------------------------------------------------------
# 1. Phasing
# ---------------------------------------------------------------------------
def figure_phasing():
    a, e, mu = P2.a_phas, P2.e_phas, P2.mu
    theta_dv1 = (P2.n_C * P2.wt) % (2 * np.pi)          # chaser polar angle at dV1

    # Transfer half-ellipse, parametrised with the eccentric anomaly E in [0, pi]
    E = np.linspace(0, np.pi, 400)
    r_tr = a * (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    t_tr = (E - e * np.sin(E)) / np.sqrt(mu / a**3)      # s from dV1

    fig = plt.figure(figsize=(12.5, 6.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0], projection="polar")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(th, np.full_like(th, H_CHASER), "--", color=C_CHASER, lw=1.6,
            label=f"Chaser parking orbit ({H_CHASER:.0f} km)")
    ax.plot(th, np.full_like(th, H_TARGET), "--", color=C_TARGET, lw=1.6,
            label=f"Target orbit ({H_TARGET:.0f} km)")
    ax.plot(theta_dv1 + nu, r_tr - RE, color=C_TRANSFER, lw=3, label="Hohmann transfer")
    ax.plot([0], [H_CHASER], "o", ms=9, color=C_CHASER, mec="white", mew=1.5, ls="none",
            label="Chaser at t = 0")
    ax.plot([theta_dv1], [H_CHASER], "^", ms=11, color=C_LIMIT, mec="white", mew=1.5, ls="none",
            label=f"ΔV₁ = {abs(P2.delta_v1_chaser) * 1000:.1f} m/s")
    ax.plot([theta_dv1 + np.pi], [H_TARGET], "v", ms=11, color=C_LIMIT, mec="white", mew=1.5,
            ls="none", label=f"ΔV₂ = {abs(P2.delta_v2_chaser) * 1000:.1f} m/s (rendezvous)")
    ax.set_ylim(330, 575)
    ax.set_yticks([400, 450, 500, 550])
    ax.set_yticklabels(["400 km", "450", "500", "550 km"], fontsize=8.5, color=C_INK2)
    ax.set_rlabel_position(100)
    ax.set_title("(a) Orbital plane, radial axis = altitude", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    wt_min = P2.wt / 60
    t_min = np.concatenate([[0, wt_min], wt_min + t_tr / 60, [wt_min + t_tr[-1] / 60 + 25]])
    h_km = np.concatenate([[H_CHASER, H_CHASER], r_tr - RE, [H_TARGET]])
    ax2.plot(t_min, h_km, color=C_TRANSFER, lw=2.4)
    ax2.axvline(wt_min, color=C_LIMIT, ls=":", lw=1.4)
    ax2.axvline(wt_min + P2.TOF / 60, color=C_LIMIT, ls=":", lw=1.4)
    ax2.text(wt_min / 2, 420, f"waiting in the parking orbit: {P2.wt / 3600:.2f} h",
             ha="center", va="center", color=C_INK2)
    ax2.text(wt_min - 6, 530, f"transfer\n{P2.TOF / 60:.1f} min", ha="right", va="center",
             color=C_INK2)
    ax2.set_xlabel("Time from start [min]")
    ax2.set_ylabel("Chaser altitude [km]")
    ax2.set_ylim(380, 570)
    ax2.set_title(f"(b) Chaser altitude (target starts {P2.delta_phi_0:.0f}° ahead)")
    fig.suptitle(f"Phasing: Hohmann transfer {H_CHASER:.0f} km → {H_TARGET:.0f} km, "
                 f"total ΔV {(abs(P2.delta_v1_chaser) + abs(P2.delta_v2_chaser)) * 1000:.1f} m/s",
                 fontsize=13, fontweight="bold")
    save(fig, "readme_phasing")


# ---------------------------------------------------------------------------
# 2. Re-entry of the stack
# ---------------------------------------------------------------------------
def figure_decay():
    t, h = P4.t_days, P4.h
    rev = np.floor(np.unwrap(np.arctan2(P4.y, P4.x)) / (2 * np.pi)).astype(int)
    t_rev, h_apo, h_peri = [], [], []
    for k in np.unique(rev):
        m = rev == k
        if m.sum() < 5:
            continue
        t_rev.append(t[m].mean())
        h_apo.append(h[m].max())
        h_peri.append(h[m].min())
    t_reentry = P4.t_decay / 86400

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, h, color=C_MUTED, lw=0.3, alpha=0.45, label="Instantaneous altitude")
    ax.plot(t_rev, h_apo, color=C_TARGET, lw=2.4, label="Apogee")
    ax.plot(t_rev, h_peri, color=C_CHASER, lw=2.4, label="Perigee")
    ax.axhline(P4.h_reentry, color=C_LIMIT, ls="--", lw=1.4, label=f"Re-entry ({P4.h_reentry} km)")
    ax.annotate(f"re-entry after {t_reentry:.1f} days\n({P4.n_rev:.0f} revolutions)",
                xy=(t_reentry, P4.h_reentry), xytext=(t_reentry * 0.55, 150),
                color=C_INK, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_INK2, lw=1.2))
    ax.set_xlabel("Time after the deorbit burn [days]")
    ax.set_ylabel("Altitude [km]")
    ax.set_ylim(80, 580)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(f"Re-entry of the chaser + Starlink stack from the "
                 f"{P4.apogee:.0f} × {P4.perigee:.0f} km ellipse\n"
                 f"NRLMSISE-00, moderate solar activity, B$_C$ = {P4.B_c:.1f} kg/m²")
    fig.tight_layout()
    save(fig, "readme_decay")


# ---------------------------------------------------------------------------
# 3. Solar activity: ADR vs natural decay
# ---------------------------------------------------------------------------
def read_scenarios():
    path = os.path.join(percorsi.DIR_RISULTATI, "rientro_scenari_solari.csv")
    adr, natural = {}, {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            val = lambda k: float(row[k].replace(",", ".")) if row[k] else None
            if row["gruppo"] == "sistema" and row["caso"] == "chaser 150 kg secco":
                adr[row["scenario"]] = val("valore_completo")          # days, full NRLMSISE-00
            if row["gruppo"] == "naturale" and row["caso"] == "A target 4 m2":
                natural[row["scenario"]] = val("valore_tabella") * 365.25   # years -> days
    return adr, natural


def figure_solar_activity():
    adr, natural = read_scenarios()
    scen = ["bassa", "moderata", "alta"]
    names = ["Low\n(F10.7 = 65)", "Moderate\n(F10.7 = 140)", "High\n(F10.7 = 250)"]
    x = np.arange(len(scen))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    b1 = ax.bar(x - w / 2 - 0.01, [adr[s] for s in scen], w, color=C_CHASER,
                label="With ADR: chaser + target, deorbit burn to 550 × 200 km")
    b2 = ax.bar(x + w / 2 + 0.01, [natural[s] for s in scen], w, color=C_TARGET,
                label="Without ADR: target left in its 550 km orbit")
    ax.set_yscale("log")
    ax.set_ylim(3, 1e5)
    for bar, s in zip(b1, scen):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15, f"{adr[s]:.1f} days",
                ha="center", va="bottom", fontsize=10, color=C_INK, fontweight="bold")
    for bar, s in zip(b2, scen):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{natural[s] / 365.25:.1f} years", ha="center", va="bottom", fontsize=10,
                color=C_INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("Solar activity (NRLMSISE-00, ECSS scenarios)")
    ax.set_ylabel("Time to re-entry at 100 km [days, log scale]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Time to re-entry of the Starlink target, with and without ADR")
    fig.tight_layout()
    save(fig, "readme_solar_activity")


if __name__ == "__main__":
    print("\nFigures for the README:")
    figure_phasing()
    figure_decay()
    figure_solar_activity()
    print(f"Output folder: {OUTDIR}")

"""Phase 4 - Atmospheric decay: propagation of the 550 x 200 km stack down to 100 km.

Equations of motion in the equatorial plane: Keplerian gravity (point-mass Earth,
no J2) plus atmospheric drag. The atmosphere co-rotates with the Earth and its
density follows the US Standard Atmosphere 1976 (piecewise exponential table from
H. D. Curtis, "Orbital Mechanics for Engineering Students", atmosphere.m).
Integrator: scipy solve_ivp, DOP853, stopped by a re-entry event at 100 km.

Importing this module runs the propagation (~10 s), because the figure scripts
reuse its results; the plot is produced only when the file is run directly.
Run:   python phase4_decay.py
"""
import os

import numpy as np
from scipy.integrate import solve_ivp

import phase3_deorbit as p3
from constants import H_REENTRY, W_EARTH

mu = p3.mu
R_earth = p3.r_earth
B_c = p3.B_c                         # kg/m^2, ballistic coefficient of the stack
w_earth = W_EARTH                    # rad/s

# --- Initial orbit: apogee of the re-entry ellipse from phase 3 ------------------------
apogee = p3.r_target - R_earth       # 550 km
perigee = p3.r_deorbit - R_earth     # 200 km
r_apog = apogee + R_earth
r_perig = perigee + R_earth
a_deorbit = (r_apog + r_perig) / 2
v_apog = np.sqrt(mu * (2 / r_apog - 1 / a_deorbit))
T_orbit = 2 * np.pi * np.sqrt(a_deorbit**3 / mu)

state_vector = [r_apog, 0, 0, v_apog]   # [x, y, vx, vy] in km and km/s, at apogee

# --- US Standard Atmosphere 1976: rho(h) = rho_i * exp(-(h - h_i) / H_i) ---------------
h_atm = np.array([   0,    25,    30,    40,    50,    60,    70,
                    80,    90,   100,   110,   120,   130,   140,
                   150,   180,   200,   250,   300,   350,   400,
                   450,   500,   600,   700,   800,   900,  1000])          # km

rho_atm = np.array([1.225,    4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
                    1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
                    2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
                    1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15])  # kg/m^3

H_atm = np.array([  7.310,  6.427,  6.546,  7.360,  8.342,  7.583,  6.661,
                    5.927,  5.533,  5.703,  6.782,  9.973, 13.243, 16.322,
                   21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
                   60.980, 65.654, 76.377, 100.587, 147.203, 208.020])   # km, scale heights


def density(h):
    """Atmospheric density [kg/m^3] at geometric altitude h [km].

    The table covers 0-1000 km; outside that range the altitude is clipped, so the
    function stays defined during the final part of the re-entry.
    """
    z = np.clip(h, 0.0, 1000.0)
    i = np.searchsorted(h_atm, z, side="right") - 1
    i = min(i, len(H_atm) - 1)
    return rho_atm[i] * np.exp(-(z - h_atm[i]) / H_atm[i])


def kepler_2d(t, state):
    """Planar two-body dynamics with drag relative to the co-rotating atmosphere."""
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    rho = density(r - R_earth)

    ax = -mu * x / r**3
    ay = -mu * y / r**3

    v_atm = [-w_earth * y, w_earth * x]                 # km/s
    v_rel = np.subtract([vx, vy], v_atm) * 1000         # m/s
    v_rel_norm = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
    v_rel_hat = v_rel / v_rel_norm

    ax_drag = -0.5 * rho * v_rel_norm**2 * v_rel_hat[0] / B_c / 1000   # km/s^2
    ay_drag = -0.5 * rho * v_rel_norm**2 * v_rel_hat[1] / B_c / 1000
    return [vx, vy, ax + ax_drag, ay + ay_drag]


h_reentry = H_REENTRY   # km


def reentry(t, state):
    """Event: altitude crosses h_reentry going down (terminates the integration)."""
    return np.sqrt(state[0]**2 + state[1]**2) - R_earth - h_reentry


reentry.terminal = True
reentry.direction = -1

# --- Integrator check: altitude after a single orbit -----------------------------------
check = solve_ivp(kepler_2d, [0, T_orbit], state_vector, method="DOP853", rtol=1e-10, atol=1e-12)
r_check = np.sqrt(check.y[0, -1]**2 + check.y[1, -1]**2)

# --- Propagation until re-entry (upper limit: 3 years, output every 60 s) -------------
t_max = 3 * 365 * 24 * 3600
t_eval = np.arange(0, t_max, 60)
solution = solve_ivp(kepler_2d, [0, t_max], state_vector, method="DOP853", rtol=1e-8, atol=1e-9,
                     t_eval=t_eval, events=reentry)

x = solution.y[0]
y = solution.y[1]
r = np.sqrt(x**2 + y**2)
h = r - R_earth
t_days = solution.t / 86400

# Revolutions from the unwrapped polar angle (t_decay / T_orbit would underestimate
# them, because the period shrinks during the decay)
n_rev = np.unwrap(np.arctan2(y, x))[-1] / (2 * np.pi)
t_decay = solution.t_events[0][0] if solution.t_events[0].size > 0 else None


def main():
    import matplotlib.pyplot as plt

    print(f"Altitude after one orbit (with drag): {r_check - R_earth:.4f} km")
    if t_decay is not None:
        print(f"Decay time from {apogee:.0f}x{perigee:.0f} km to {h_reentry:.0f} km: "
              f"{t_decay / 86400:.2f} days ({t_decay / 3600:.1f} hours)")
        print(f"Number of revolutions before re-entry: {n_rev:.0f}")
    else:
        print(f"No re-entry within {t_max / 86400:.0f} days: final altitude {h[-1]:.2f} km")
    print(f"Ballistic coefficient used: {B_c:.2f} kg/m^2")

    plt.figure(figsize=(10, 5))
    plt.plot(t_days, h, linewidth=0.5)
    plt.axhline(h_reentry, color="r", linestyle="--", linewidth=1,
                label=f"Re-entry ({h_reentry:.0f} km)")
    plt.xlabel("Time [days]")
    plt.ylabel("Altitude [km]")
    plt.title(f"Orbital decay - initial orbit {apogee:.0f}x{perigee:.0f} km, "
              f"B$_C$={B_c:.1f} kg/m$^2$")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "decay_altitude.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

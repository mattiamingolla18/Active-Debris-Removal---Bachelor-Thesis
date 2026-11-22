"""Phase 2 - Phasing: Hohmann transfer of the chaser from 400 km to the target at 550 km.

The chaser waits in its parking orbit until the phase angle with the target equals
the lead angle of a Hohmann transfer, then performs two impulsive burns:
dV1 (prograde, at perigee) and dV2 (circularisation, at apogee).

Model: point-mass Earth, coplanar circular equatorial orbits, impulsive burns.
Run:   python phase2_phasing.py
"""
import numpy as np

from constants import (CHASER_MASS, G0, H_CHASER, H_TARGET, INITIAL_PHASE_DEG,
                       ISP, MU, R_EARTH, TARGET_MASS)

# Short aliases (also used by the figure scripts)
mu = MU
r_earth = R_EARTH
g = G0
Isp = ISP
mass_chaser = CHASER_MASS
starlink_mass = TARGET_MASS

r_target = R_EARTH + H_TARGET   # km
r_chaser = R_EARTH + H_CHASER   # km

# --- Hohmann transfer ellipse --------------------------------------------------
e_phas = (r_target - r_chaser) / (r_target + r_chaser)
a_phas = (r_chaser + r_target) / 2
TOF = np.pi * np.sqrt(a_phas**3 / mu)          # s, time of flight (half ellipse)

# --- Lead angle ----------------------------------------------------------------
w_t = np.sqrt(mu / r_target**3)                # rad/s, target angular velocity
theta_t = w_t * TOF                            # angle travelled by the target during the transfer
phi = np.pi - theta_t                          # required lead angle of the target at dV1

# --- Mean motions --------------------------------------------------------------
T_chaser = 2 * np.pi * np.sqrt(r_chaser**3 / mu)
n_C = 2 * np.pi / T_chaser
T_Target = 2 * np.pi * np.sqrt(r_target**3 / mu)
n_T = 2 * np.pi / T_Target

# --- Waiting time in the parking orbit -------------------------------------------
delta_phi_0 = INITIAL_PHASE_DEG                # deg
phi_difference = np.radians(delta_phi_0) - phi
if phi_difference < 0:
    phi_difference += 2 * np.pi
wt = phi_difference / (n_C - n_T)              # s

# --- Burns ---------------------------------------------------------------------
v_circ = np.sqrt(mu / r_chaser)
v_chaser_perigee = np.sqrt(2 * mu * (1 / r_chaser - 1 / (2 * a_phas)))
delta_v1_chaser = v_chaser_perigee - v_circ    # km/s

v_chaser_apogee = np.sqrt(2 * mu * (1 / r_target - 1 / (2 * a_phas)))
v_circ_2 = np.sqrt(mu / r_target)
delta_v2_chaser = v_circ_2 - v_chaser_apogee   # km/s

# --- Propellant (Tsiolkovsky) ----------------------------------------------------
total_delta_v_chaser = (abs(delta_v1_chaser) + abs(delta_v2_chaser)) * 1000   # m/s
final_mass_chaser = mass_chaser * np.exp(-total_delta_v_chaser / (Isp * g))
propellant_mass = mass_chaser - final_mass_chaser


def main():
    print(f"Initial phase angle: {delta_phi_0:.2f} deg")
    print(f"Waiting time before phasing: {int(wt) // 3600:d} h {int(wt % 3600) // 60:d} min "
          f"{wt % 60:.2f} s")
    print(f"Transfer time of flight: {TOF:.2f} s")
    print(f"Delta-V1 (chaser): {delta_v1_chaser:.6f} km/s")
    print(f"Delta-V2 (chaser): {delta_v2_chaser:.6f} km/s")
    print(f"Propellant burned for phasing: {propellant_mass:.4f} kg")
    print(f"Chaser mass after phasing: {final_mass_chaser:.4f} kg")


if __name__ == "__main__":
    main()

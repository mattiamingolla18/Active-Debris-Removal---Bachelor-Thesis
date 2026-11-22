"""Phase 3 - Deorbit burn: lowering the perigee of the captured target to 200 km.

After the rendezvous the chaser is assumed to be rigidly attached to the Starlink
satellite. A single retrograde burn at 550 km puts the stack on a 550 x 200 km
ellipse; the atmosphere then completes the re-entry (phase 4).

Model: point-mass Earth, impulsive burn, capture treated as instantaneous.
Run:   python phase3_deorbit.py
"""
import numpy as np

import phase2_phasing as p2
from constants import AREA_CHASER, AREA_TARGET, CD, H_DEORBIT_PERIGEE

mu = p2.mu
r_earth = p2.r_earth
g = p2.g
Isp = p2.Isp
starlink_mass = p2.starlink_mass
r_target = p2.r_target
r_deorbit = r_earth + H_DEORBIT_PERIGEE      # km, perigee of the re-entry ellipse

Cd_target = CD
Cd_chaser = CD
frontal_area_starlink = AREA_TARGET
frontal_area_chaser = AREA_CHASER

# --- Mass of the chaser + target stack before the burn ----------------------------
final_chaser_mass = p2.final_mass_chaser
total_mass = final_chaser_mass + starlink_mass

# --- Re-entry ellipse and deorbit burn ----------------------------------------------
e_deorbit = (r_target - r_deorbit) / (r_target + r_deorbit)
a_deorbit = (r_deorbit + r_target) / 2
v1 = np.sqrt(mu / r_target)                               # circular velocity at 550 km
v2 = np.sqrt(2 * mu / r_target - mu / a_deorbit)          # apogee velocity on the ellipse
delta_v3 = v1 - v2                                        # km/s, retrograde burn

m_prop = total_mass * (1 - np.exp(-abs(delta_v3) * 1000 / (Isp * g)))
total_propellant_burned = p2.propellant_mass + m_prop
mass_chaser_after_burned = final_chaser_mass - m_prop
total_mass_after_burn = mass_chaser_after_burned + starlink_mass

# --- Ballistic coefficient of the stack: B_C = m / (Cd_t*A_t + Cd_c*A_c) -------------
Cd_A = Cd_target * frontal_area_starlink + Cd_chaser * frontal_area_chaser
B_c = total_mass_after_burn / Cd_A                        # kg/m^2

total_delta_v_mission = abs(p2.delta_v1_chaser) + abs(p2.delta_v2_chaser) + abs(delta_v3)


def main():
    print(f"Stack mass before the deorbit burn: {total_mass:.2f} kg")
    print(f"Delta-V3 (deorbit burn): {delta_v3:.6f} km/s")
    print(f"Propellant burned for deorbiting: {m_prop:.2f} kg")
    print(f"Total propellant burned: {total_propellant_burned:.2f} kg")
    print(f"Chaser mass after deorbiting: {mass_chaser_after_burned:.2f} kg")
    print(f"Stack mass after the deorbit burn: {total_mass_after_burn:.2f} kg")
    print(f"Ballistic coefficient after deorbiting: {B_c:.2f} kg/m^2")

    print("\n--- Delta-V budget ---")
    print(f"Delta-V1 (phasing, departure from 400 km):  {abs(p2.delta_v1_chaser) * 1000:8.3f} m/s")
    print(f"Delta-V2 (phasing, insertion at 550 km):    {abs(p2.delta_v2_chaser) * 1000:8.3f} m/s")
    print(f"Delta-V3 (deorbit burn at 550 km):          {abs(delta_v3) * 1000:8.3f} m/s")
    print(f"Total mission Delta-V:                      {total_delta_v_mission * 1000:8.3f} m/s")


if __name__ == "__main__":
    p2.main()
    print()
    main()

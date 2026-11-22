"""Mission parameters and physical constants shared by every phase.

Change a value here and all phases (and all figures) are recomputed consistently.
"""

# --- Earth -------------------------------------------------------------------
R_EARTH = 6378.1363       # km, equatorial radius
MU = 398600.44            # km^3/s^2, gravitational parameter
W_EARTH = 7.2921e-5       # rad/s, rotation rate (the atmosphere co-rotates with it)
G0 = 9.81                 # m/s^2, standard gravity (rocket equation)

# --- Orbits ------------------------------------------------------------------
H_CHASER = 400.0          # km, circular parking orbit of the chaser
H_TARGET = 550.0          # km, circular orbit of the decommissioned Starlink
H_DEORBIT_PERIGEE = 200.0 # km, perigee of the re-entry ellipse after the deorbit burn
H_REENTRY = 100.0         # km, conventional re-entry altitude (Karman line)
INITIAL_PHASE_DEG = 45.0  # deg, initial phase angle of the target ahead of the chaser

# --- Spacecraft --------------------------------------------------------------
CHASER_MASS = 400.0       # kg, chaser wet mass at the start of the mission
TARGET_MASS = 295.0       # kg, Starlink satellite mass
ISP = 300.0               # s, chaser specific impulse
CD = 2.2                  # drag coefficient (same value for target and chaser)
AREA_TARGET = 4.0         # m^2, frontal area of the target
AREA_CHASER = 2.0         # m^2, frontal area of the chaser

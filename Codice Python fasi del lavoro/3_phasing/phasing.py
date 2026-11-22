import os
import sys
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","Comuni"))
import percorsi
import config_missione as cfg

r_earth=6378.1363
r_target = 550+ r_earth # 550km of altitude
r_chaser = 400+ r_earth # 400km of altitude

mu = 398600.44 # km^3/s^2
starlink_mass=cfg.M_TARGET
g=9.81 # m/s^2
Isp=cfg.ISP # seconds
 
## Hohmann Transfer orbit:
e_phas = (r_target - r_chaser) / (r_target + r_chaser)   # fix: era r_target-r_target
a_phas = (r_chaser + r_target) / 2
TOF = np.pi * (np.sqrt(a_phas**3 / mu))
 
## Angular velocity of the Tgt:
w_t = np.sqrt(mu / (r_target**3))
 
## Angular distance of the Tgt during the chaser transfer:
theta_t = w_t * TOF
 
## Lead angle between Chaser and Tgt:
phi = np.pi - theta_t
 
## Mean motion Chaser:
T_chaser = (2 * np.pi * (np.sqrt(r_chaser**3))) / (np.sqrt(mu))
n_C = (2 * np.pi) / T_chaser
 
## Mean motion Target:
T_Target = (2 * np.pi * (np.sqrt(r_target**3))) / (np.sqrt(mu))
n_T = (2 * np.pi) / (T_Target)
 
delta_phi_0= 45# initial phase angle in degrees
print(f"Initial Phase Angle: {delta_phi_0:.2f} degrees")
phi_difference= (np.radians(delta_phi_0) - phi) # difference between initial phase angle and lead angle
if phi_difference < 0:
    phi_difference = phi_difference+ (2 * np.pi) 

## Wait time:
wt=phi_difference/(n_C-n_T)

print(f"Wait Time for Phasing: {int(wt)//3600:d} hours, {int(wt%3600)//60:d} minutes, {wt%60:.2f} seconds")

## Delta-V1 for the chaser:
v_circ = np.sqrt(mu/r_chaser)
v_chaser_perigee = np.sqrt(2*mu*(1/r_chaser-1/(2*a_phas)))
delta_v1_chaser=v_chaser_perigee-v_circ
print(f"Delta-V1 for the chaser: {delta_v1_chaser:.6f} km/s")

## Delta-V2 for the chaser:
v_chaser_apogee= np.sqrt(2*mu*(1/r_target-1/(2*a_phas)))
v_circ_2= np.sqrt(mu/r_target)
delta_v2_chaser=v_circ_2 - v_chaser_apogee
print(f"Delta-V2 for the chaser: {delta_v2_chaser:.6f} km/s")

## Total Delta-V for the maneuver:
total_delta_v_chaser=(np.abs(delta_v1_chaser) + np.abs(delta_v2_chaser))*1000 #km/s -> m/s

## Massa al lancio, ricavata a ritroso dalla massa a secco.
## Il Delta-V3 dipende solo dalle orbite (deorbiting.py lo ricalcola con la stessa formula).
r_perigeo_rientro=200+r_earth
a_ellisse_rientro=(r_perigeo_rientro+r_target)/2
dv3_dimensionamento=(np.sqrt(mu/r_target)-np.sqrt(2*mu/r_target-mu/a_ellisse_rientro))*1000 #m/s
mass_chaser=cfg.massa_al_lancio(cfg.M_SECCO_CHASER,starlink_mass,total_delta_v_chaser,
                                dv3_dimensionamento,Isp,g,cfg.MARGINE)
print(f"Launch mass of the chaser: {mass_chaser:.4f} kg")

mass_ratio=np.exp(-total_delta_v_chaser/(Isp*g))
final_mass_chaser=mass_chaser*mass_ratio
propellent_mass=mass_chaser-final_mass_chaser

print(f"Propellant burned for Phasing: {propellent_mass:.4f} kg")
print(f"Final Mass of Chaser after Phasing:  {final_mass_chaser:.4f} kg")

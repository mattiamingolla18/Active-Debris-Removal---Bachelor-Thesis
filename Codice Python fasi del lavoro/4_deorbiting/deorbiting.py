import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","Comuni"))
import percorsi
from phasing import *
import numpy as np
import config_missione as cfg
r_earth=6378.1363
r_target = 550+ r_earth # 550km of altitude
r_chaser = 400+ r_earth # 400km of altitude
r_deorbit =200+ r_earth # 200km of altitude, perigeo dell'orbita di rientro (coerente con rientro.py)

mu = 398600.44 # km^3/s^2
g=9.81 # m/s^2
Isp=cfg.ISP # seconds
starlink_mass=cfg.M_TARGET
final_chaser_mass=final_mass_chaser # massa del chaser dopo il phasing, importata dalla fase 2
total_mass=final_chaser_mass+starlink_mass
print(f"Total mass of the system: {total_mass:.2f} kg")

Cd_target=cfg.C_D
Cd_chaser=Cd_target
frontal_area_starlink=cfg.A_TARGET
frontal_area_chaser=cfg.A_CHASER

## Hohmann Transfer orbit:
e_phas = (r_target - r_deorbit) / (r_target + r_deorbit)   
a_phas = (r_deorbit + r_target) / 2
v1= np.sqrt(mu / r_target)
v2= np.sqrt((2*mu/r_target) - (mu/a_phas))
delta_v3 = v1 - v2 ## Retrograde burn to deorbit the target satellite
print(f"Delta V3 for Hohmann transfer: {delta_v3:.6f} km/s")

m_prop=total_mass * (1 - np.exp(-abs(delta_v3)*1000/(Isp*g))) # delta_v3 convertito da km/s a m/s
print(f"Propellant burned for the deorbiting: {m_prop:.2f} kg")


#### Printa i valori di propellente e delta-v per la manovra di deorbiting###
total_propellant_burned = propellent_mass + m_prop
print(f"Total propellant burned: {total_propellant_burned:.2f} kg")

mass_chaser_after_burned = final_chaser_mass - m_prop
print(f"Mass of chaser after deorbiting: {mass_chaser_after_burned:.2f} kg")

## Massa del sistema chaser+target che affronta il decadimento (dopo il burn di deorbiting):
total_mass_after_burn = mass_chaser_after_burned + starlink_mass
print(f"Total mass of the system after deorbiting burn: {total_mass_after_burn:.2f} kg")

## Coefficiente balistico B_C = m / (C_D * A), con l'area di riferimento data dalla somma
## dei contributi di target e chaser: C_D_t*A_t + C_D_c*A_c
Cd_A = Cd_target*frontal_area_starlink + Cd_chaser*frontal_area_chaser
B_c= total_mass_after_burn / Cd_A
print(f"Ballistic Coefficient after deorbiting: {B_c:.2f} kg/m^2")

## B_C del sistema per un chaser di massa a secco nota, con la riserva di propellente
def B_c_sistema(m_secco,A_target=frontal_area_starlink):
    dv1=abs(delta_v1_chaser)*1000
    dv2=abs(delta_v2_chaser)*1000
    dv3=abs(delta_v3)*1000
    m_lancio=cfg.massa_al_lancio(m_secco,starlink_mass,dv1+dv2,dv3,Isp,g,cfg.MARGINE)
    m_fine=cfg.bilancio_in_avanti(m_lancio,starlink_mass,dv1,dv2,dv3,Isp,g)[3]
    return (m_fine+starlink_mass)/(Cd_target*(A_target+cfg.area_chaser(m_secco)))

#### Riepilogo delta-V della missione ####
total_delta_v_mission = np.abs(delta_v1_chaser) + np.abs(delta_v2_chaser) + np.abs(delta_v3)
print("\n--- Delta-V budget ---")
print(f"Delta-V1 (phasing, departure from 400 km):  {np.abs(delta_v1_chaser)*1000:8.3f} m/s")
print(f"Delta-V2 (phasing, insertion at 550 km):    {np.abs(delta_v2_chaser)*1000:8.3f} m/s")
print(f"Delta-V3 (deorbiting burn at 550 km):       {np.abs(delta_v3)*1000:8.3f} m/s")
print(f"Total mission Delta-V:                      {total_delta_v_mission*1000:8.3f} m/s")


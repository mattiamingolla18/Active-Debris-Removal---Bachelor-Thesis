import numpy as np
from atmosfera_nrlmsise import importa_pymsis
pymsis=importa_pymsis()

## Modello completo: NRLMSISE-00 a ogni passo con il tempo locale reale, nelle stesse condizioni
## di FreeFlyer (stato iniziale subito dopo il Delta-V3, stessa epoca, atmosfera co-rotante).
mu=398600.44; R_earth=6378.1363; w_earth=7.2921e-5 # stesse costanti delle fasi 2-4
epoch0=np.datetime64('2020-01-01T07:12:03.278') # istante del Delta-V3 in FreeFlyer (UTC)
state0=[-4347.24654310571,-5394.48967637143,5.82898990349380,-4.69739630604348] # [x,y,vx,vy] km, km/s
h_reentry=100.0

def gmst_deg(t64):
    jd=(t64-np.datetime64('2000-01-01T12:00:00')).astype('timedelta64[ms]').astype(float)/86400e3+2451545.0
    return (280.46061837+360.98564736629*(jd-2451545.0))%360.0

def make_rhs(f107,ap,B_c,j2=0.0):
    aps=[[ap]*7]
    def rhs(t,s):
        x,y,vx,vy=s; r=np.hypot(x,y); h=r-R_earth
        t64=epoch0+np.timedelta64(int(round(t*1000)),'ms')
        lon=(np.degrees(np.arctan2(y,x))-gmst_deg(t64))%360.0 # longitudine terrestre (orbita equatoriale: lat = 0)
        rho=pymsis.calculate(t64,lon,0.0,max(h,80.0),f107s=f107,f107as=f107,aps=aps,version=0)[...,0].item()
        v_rel=np.array([vx+w_earth*y,vy-w_earth*x])*1e3 # m/s, atmosfera co-rotante
        a_d=-0.5*rho*np.hypot(*v_rel)*v_rel/B_c/1e3 # km/s^2
        a_j2=-1.5*j2*mu*R_earth**2/r**5
        return [vx,vy,-mu*x/r**3+a_d[0]+a_j2*x,-mu*y/r**3+a_d[1]+a_j2*y]
    return rhs

def reentry(t,s): return np.hypot(s[0],s[1])-R_earth-h_reentry
reentry.terminal=True; reentry.direction=-1

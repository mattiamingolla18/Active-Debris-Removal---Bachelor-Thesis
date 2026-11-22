# Active Debris Removal of a Starlink Satellite in LEO

**Bachelor's thesis in Aerospace Engineering, University of Padova**

A Python and FreeFlyer simulation of an Active Debris Removal (ADR) mission. A small chaser
(150 kg dry) intercepts a decommissioned Starlink v1.5 satellite at 550 km, captures it and lowers
the perigee of the stack to 200 km. Atmospheric drag then brings both down in about **16 days**
at moderate solar activity. Left alone, the same satellite would stay in orbit for about **4 years**,
and for more than 45 years at low solar activity.

![FreeFlyer view of the rendezvous](Grafici%20e%20immagini/readme/freeflyer_rendezvous.jpg)
*FreeFlyer view from above the North Pole: the chaser reaches the Starlink target in its 550 km orbit.*

## Mission scenario

| Parameter | Value |
|---|---|
| Target | Starlink v1.5, 275 kg dry mass, 4 m² cross-section, circular orbit at 550 km |
| Chaser | 150 kg dry, 171.3 kg at launch (10% propellant reserve included), I<sub>sp</sub> = 300 s, parking orbit at 400 km |
| Geometry | Coplanar equatorial orbits, target 45° ahead of the chaser at t = 0 |
| Manoeuvres | Hohmann phasing 400 → 550 km, capture, retrograde burn to a 550 × 200 km ellipse |
| Atmosphere | NRLMSISE-00 (via `pymsis`), three ECSS solar-activity scenarios, co-rotating with the Earth |
| Drag | C<sub>D</sub> = 2.2; chaser area from a tumbling cube of equal mass (mean area = surface / 4) |

The chaser is sized backwards: starting from its dry mass and the ΔV budget, the rocket equation
gives the launch mass, including a 10% reserve of the propellant consumed.

## Results (moderate solar activity)

| Quantity | Value |
|---|---|
| Waiting time in the parking orbit | 5 h 35 min |
| Transfer time (ΔV<sub>1</sub> → ΔV<sub>2</sub>) | 47.1 min |
| ΔV<sub>1</sub> / ΔV<sub>2</sub> (phasing) | 41.85 / 41.62 m/s |
| ΔV<sub>3</sub> (deorbit burn at 550 km) | 98.93 m/s |
| **Total ΔV** | **182.4 m/s** |
| Propellant used | 19.4 kg (4.8 kg phasing + 14.6 kg deorbit) |
| Ballistic coefficient of the stack | 35.3 kg/m² |
| **Time to re-entry (100 km)** | **15.5 days** (247 revolutions) with tabulated NRLMSISE-00; **16.0 days** with NRLMSISE-00 evaluated at every step |

![Phasing](Grafici%20e%20immagini/readme/readme_phasing.png)

![Re-entry of the stack](Grafici%20e%20immagini/readme/readme_decay.png)

Solar activity changes the atmospheric density at these altitudes by an order of magnitude, so the
re-entry time was computed for three scenarios. The deorbit burn turns years into days in all of them:

![Re-entry with and without ADR](Grafici%20e%20immagini/readme/readme_solar_activity.png)

## Validation against FreeFlyer

The mission is also modelled in FreeFlyer 7.10 (`FREEFLYER SIMULATION/`), using the ΔVs computed in
Python as manoeuvre inputs, while FreeFlyer propagates the trajectory and the re-entry on its own.
The Python values below use NRLMSISE-00 evaluated at every step, like FreeFlyer.

| Check | Python | FreeFlyer |
|---|---|---|
| Re-entry time, low solar activity | 31.987 days | 31.999 days |
| Re-entry time, moderate solar activity | 16.026 days | 16.025 days |
| Re-entry time, high solar activity | 8.866 days | 8.863 days |
| Apsidal precession with J2 | 8.1692 °/day | 8.1685 °/day |

J2 is off in the nominal case and can be switched on (`ADR_J2=1`). With the perigee held at 200 km it
changes the re-entry time by 2% at most. It matters for the rendezvous instead: a phasing plan
computed without J2 and flown with J2 misses the target by about 8.5 km, so `6_J2/effetto_J2.py`
also computes a corrected plan.

## Model assumptions

- Point-mass Earth by default (J2 optional), coplanar equatorial orbits
- Impulsive burns with constant specific impulse
- Capture is not modelled dynamically: after the rendezvous, chaser and target are a single body
- Constant drag coefficient; the target's attitude is represented by its cross-section, with a sensitivity study on area and mass in `1_target/analisi_target.py`
- Equations of motion integrated with `scipy.integrate.solve_ivp` (DOP853), stopped by a re-entry event at 100 km

## Repository structure

```
Codice Python fasi del lavoro/       Python code, one folder per mission phase
  Comuni/                            mission parameters (config_missione.py) and paths
  1_target/                          target ballistic coefficient, natural decay, drag-sail comparison
  2_chaser/                          backward sizing of the chaser, sweeps on dry mass and Isp
  3_phasing/                         Hohmann phasing: waiting time, ΔV1, ΔV2
  4_deorbiting/                      capture, deorbit burn, ballistic coefficient
  5_rientro/                         re-entry propagation, solar scenarios, NRLMSISE-00 tables
  6_J2/                              effect of J2 on precession, rendezvous and re-entry
  7_validazione_freeflyer/           FreeFlyer runs for the solar scenarios and the J2 check
  8_grafici/                         thesis, presentation and README figures
  controlli/                         regression tests (test_riferimento.py)
  Risultati/                         CSV results
FREEFLYER SIMULATION/                FreeFlyer mission plan, solar-flux files and results
Grafici e immagini/                  all figures (thesis figures are labelled in Italian)
```

Code comments, folder names and thesis figures are in Italian, as in the thesis. A full description
of the files in Italian is in [README_IT.md](README_IT.md).

## How to run

```bash
cd "Codice Python fasi del lavoro"
python -m pip install -r requirements.txt
python 4_deorbiting/deorbiting.py               # phasing + deorbit: ΔV, masses, ballistic coefficient
python 5_rientro/rientro.py                     # re-entry propagation
python 5_rientro/rientro_scenari_solari.py      # three solar-activity scenarios (~3 min)
python 8_grafici/grafici_readme.py              # figures of this README
python controlli/test_riferimento.py            # regression checks
```

Every script finds the other modules through `Comuni/percorsi.py`, so it can be launched from any
folder. Scenarios can be changed without editing the code, for example
`ADR_ATMOSFERA=alta python 5_rientro/rientro.py` (`bassa`, `moderata`, `alta`) or
`ADR_J2=1 python 5_rientro/rientro.py`. CSV files use a semicolon separator and a decimal comma, so
they open directly in Excel with Italian settings.

The FreeFlyer mission plan writes its outputs with absolute paths on the author's machine: update the
`Maneuver` and `Report` paths before running it elsewhere.

## References

- H. D. Curtis, *Orbital Mechanics for Engineering Students*
- L. Savioli, *Analysis of innovative scenarios and key technologies to perform active debris removal with satellite modules*, PhD thesis, University of Padova (CISAS), 2015
- A. M. Masserini, *Design and optimisation of an active debris removal service for large satellite constellations*, MSc thesis, Politecnico di Milano, 2020
- C. Steimle, J. Marie, S. Woicke, L. Metrailler, *ClearSpace-1 In-Orbit Demonstration Mission*, SpaceOps 2025
- M. Ballard, G. Song, T. Zhu, *Chain Reactions in Space: Analyzing the Impact of Satellite Collisions and Debris Accumulation*, arXiv:2512.22429, 2025
- V. Ray et al., *The impact of space weather on very low Earth orbit (VLEO) satellites*, 2022
- Y. Baruah et al., *The Loss of Starlink Satellites in February 2022*, Space Weather 22(4), 2024
- J. McDowell, *General Catalog of Artificial Space Objects* (Starlink v1.5 masses)

## Author

**Mattia Mingolla** – BSc in Aerospace Engineering, University of Padova
[LinkedIn](https://www.linkedin.com/in/mattia-mingolla/) · mattiamingollawork@gmail.com

Thesis (in Italian): *Simulazione di Active Debris Removal (ADR) in orbita LEO: Intercettazione e Deorbiting di un satellite Starlink*

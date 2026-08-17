# Citations & Supporting References

Drift-Sense – SEMICON India Hackathon 2026 (Applied Materials)

This document lists the public sources used to justify layout geometry, SEM noise modelling, and the classical localization algorithm. All citations correspond to the references used in the project presentation.

---

## 1. Die-Layout Generation

### DRAM / 6F²
- Schloesser et al., *6F² buried wordline DRAM cell for 40 nm and beyond*, IEDM 2008.  
  https://ieeexplore.ieee.org/document/4796820
- US 8,519,462 – Intel, *6F² DRAM cell*.  
  https://patents.google.com/patent/US8519462
- US 6,545,904 – Micron, *6F² DRAM array*.  
  https://www.freepatentsonline.com/6545904.html
- US 7,349,232 B2 – Micron, *6F² DRAM Cell Design with 3F-Pitch Folded Digitline Sense Amplifier* (oblique active-area geometry).  
  https://patents.google.com/patent/US7349232B2

### FinFET 6T-SRAM
- *FinFET 6T-SRAM All-Digital Compute-in-Memory…*, Micromachines 2023.  
  https://www.mdpi.com/2072-666x/14/8/1535
- US 11,342,337 B2 – *Structure and method for FinFET SRAM*.  
  https://patents.google.com/patent/US11342337B2
- US 9,012,287 B2 – *Cell Layout for SRAM FinFET Transistors* (orthogonal fin/gate cross-point).  
  https://patents.google.com/patent/US9012287
- Berkeley FinFET SRAM layout technical report (EECS-2006-138).  
  https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-138.pdf

### BEOL Interconnect
- imec – *Semi-damascene interconnects with fully self-aligned vias at 18 nm metal pitch*.  
  https://www.imec-int.com/en/articles/imec-demonstrates-semi-damascene-interconnects-fully-self-aligned-vias-18nm-metal-pitch

---

## 2. SEM Noise, Charging, Drift & Contamination

- Timischl et al. (2012), *A statistical model of signal-noise in scanning electron microscopy*, Scanning – Poisson + Gaussian (shot + readout) model.
- Jin et al. (2015), *Correction of image drift and distortion in a scanning electron microscopy*, Journal of Microscopy – stage drift and line-scan jitter.
- Muller et al. (2006), *Room design for high-performance electron microscopy*, Ultramicroscopy – AC-mains electromagnetic pickup.
- Postek & Vladár, *Does Your SEM Really Tell the Truth? Part 4: Charging and its Mitigation* (NIST).  
  https://www.nist.gov/publications/does-your-sem-really-tell-truth-how-would-you-know-part-4-charging-and-its-mitigation
- NIST notes on hydrocarbon contamination in SEM.  
  https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=914028
- Reimer, *Scanning Electron Microscopy* (Springer) – SE yield, edge brightening, contamination.
- Goldstein et al., *Scanning Electron Microscopy and X-ray Microanalysis* – topographic / edge contrast.

---

## 3. Classical Localization (Normalized Cross-Correlation)

- Lewis, *Fast Normalized Cross-Correlation* (classic NCC reference).  
  https://www.academia.edu/download/30514516/10.1.1.21.6062.pdf
- US 6,399,953 B1 – SEM patent using the normalized correlation coefficient method for wafer feature matching.
- US 8,089,612 B2 – Position detection apparatus using coarse correlation followed by local refinement.
- OpenCV `matchTemplate` documentation (implements the same family of methods).

---

## 4. Roadmaps & Problem Specification

- IEEE International Roadmap for Devices and Systems (IRDS).  
  https://irds.ieee.org/
- Drift-Sense / SEMICON India Hackathon 2026 problem statement (10× zoom, ±20 % scale, 1–3° rotation, PR-vs-noise protocol, 1/3/5 px success, timing requirements).

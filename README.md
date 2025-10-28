# hybrid-system-identification
MATLAB code for “Learning Hybrid Dynamics via Convex Optimizations” (arXiv:2509.24157)

This repository contains the MATLAB implementation accompanying the paper:

> **Kaito Iwasaki**, Maani Ghaffari, and Anthony Bloch  
> *Learning Hybrid Dynamics via Convex Optimizations* (arXiv:2509.24157)  
> [https://arxiv.org/abs/2509.24157](https://arxiv.org/abs/2509.24157)

---

## Overview
This repository provides MATLAB scripts used for identifying **switching and hybrid dynamical systems** using **bilevel convex optimization**.  
The methods combine mode assignment via convex relaxations and system identification for polynomial and linear switching models.

---

## Code Structure
| File | Description |
|------|--------------|
| `searchMode.m` | Solves for relaxed mode assignments using convex optimization |
| `searchMatrix.m` | Identifies system matrices for each mode given mode assignments |
| `searchMode_moment.m` | Moment-based SDP version of the mode search |
| `searchNetwork.m` | Mode identification for network dynamical systems |
| `searchPoly.m` / `searchPoly_old.m` | Polynomial system identification (quartic and general cases) |
| `searchSwitchingSurface_softmargin.m` | Soft-margin formulation for learning switching surfaces |
| `spot_ccm_1norm_sgd_linear_switching_ho.m` | Hybrid oscillator experiment (linear) |
| `spot_ccm_1norm_sgd_network_switching.m` | Switching network dynamics example |
| `spot_ccm_1norm_sgd_polynomial_switching.m` | Polynomial hybrid dynamics example |
| `liner_ho_switching_damping_construct.m` | Data generation for switching harmonic oscillator |
| `network_switching_construct.m` | Data generation for switching network systems |
| `quartic_oscillator_switching_construct.m` | Data generation for polynomial oscillator |
| `predict_vel_from_modes_poly.m` | Predicts velocities from identified polynomial models |

---

## Requirements
- MATLAB **R2024a** or later  
- [Spotless Toolbox](https://github.com/spot-toolbox/spotless)  
- [MOSEK Optimizer](https://www.mosek.com/downloads/) (academic license)

---

## How to Run
1. Open MATLAB and add all files to your path.
2. Run one of the construct.m files, e.g.: linear_ho_switching_damping_construct.m
3. Run one of the main scripts, e.g.: spot_ccm_1norm_sgd_linear_switching_ho.m

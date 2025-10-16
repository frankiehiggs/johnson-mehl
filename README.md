# Johnson-Mehl and related tessellations

This repository contains the code and data for several experiments and figures in the paper [Random coverage from within with variable radii, and Johnson-Mehl cover times](https://arxiv.org/abs/2405.17687) (Penrose and Higgs, 2025+).

It is divided into several folders.

---

## limiting-distribution

The code in here computes the cover time and other statistics for the Johnson-Mehl model inside $[0,1]^2$ to a high precision.

It also plots a diagram containing the empirical distribution of the samples (or rather, if $T_\rho$ is the coverage time when the arrival rate is $\rho$, it plots the empirical distribution of $\pi \rho T_\rho^3 - 2\log \rho - 4\log \log \rho$), as well as the limit for this quantity given in [Penrose and Higgs (2025+)](https://arxiv.org/abs/2405.17687), and the predicted limit given in [Chiu (1995)](https://doi.org/10.2307/1427927). Chiu's limit is valid when points are also placed outside the square, but is not the limiting distribution of our simulated data because of boundary effects (we only place points _inside_ the square).

To produce `N` samples with arrival rate `rho` and plot a diagram, run
```
python fast_jm.py rho N
```

It saves the generated samples to a file in `data/`. It saves two other interesting thresholds which are computed as intermediate steps in the simulation: the time the last point arrives in a vacant region (saved to `rho1234-final-arrivals.csv`), and the first time after this final arrival at which there are no isolated balls (saved to `rho1234-iso-thresholds.csv`). It also saves the total number of points which arrive in an uncovered region.

---

## pareto-video-code

This makes videos of the spherical Poisson Boolean model with heavy-tailed radii. As the video progresses, the radii get more and more heavy-tailed, and you can see a single ball taking over almost all of the region.

It can be run just by calling,
```
./generate.sh 1000
```
where the number is the intensity of the point process.

This video and the other diagrams below use the algorithm described by Hervé Moulinec in [A simple and fast algorithm for computing discrete Voronoi, Johnson-Mehl or Laguerre diagrams of points](https://www.sciencedirect.com/science/article/abs/pii/S0965997822000618).

---


## colourful-diagrams

The code in here wasn't used in the paper, but it produces very attractive tessellations from the spherical Poisson Boolean model (SPBM) with random radii, or the Johnson-Mehl model.

The user can choose various distributions for the radii in the SPBM, either a constant, uniform on $[0,1]$, exponential, discrete $p \delta_a + (1-p)\delta_b$, or Pareto distributed.

For example:
```
python draw_spbm.py constant
python draw_spbm.py uniform
python draw_spbm.py exponential
python draw_spbm.py discrete 1 3 0.67 0.33
python draw_spbm.py pareto 2.5 1.0
```

To draw a Johnson-Mehl tessellation just uncomment the relevant lines inside the `if __name__=='__main__'` block.

---

## notebooks

The main two interesting files in here are `jm-picture-with-boundaries.ipynb` and `spbm-picture-with-boundaries.ipynb`. These generate the illustrations of a Johnson-Mehl and SPBM tessellation in Figure 1 of the published version of Penrose and Higgs (2025+).

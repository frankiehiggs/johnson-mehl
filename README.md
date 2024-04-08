# johnson-mehl
Some experiments to test the predicted limiting CDFs for a Johnson-Mehl tessellation in a 2d box.

It also plots a diagram containing the empirical distribution of the samples (or rather, if $T_\rho$ is the coverage time when the arrival rate is $\rho$, it plots the empirical distribution of $\pi \rho T_\rho^3 - 2\log \rho - 4\log \log \rho$), as well as the limit for this quantity given in Penrose (2024+), and the predicted limit given in Chiu (1995). Chiu's limit is valid when points are also placed outside the square, but is _not_ the limiting distribution of our simulated data because of boundary effects (we only place points _inside_ the square). Penrose and Chiu use different transformations of $T$ to get a quantity which converges weakly to some limiting random variable, so I had to pick one transformation to apply to the data. I used Penrose's, so the cdf taken from Chiu's paper moves about a bit because its prediction for the distribution of Penrose's quantity also depends on $\rho$. I could have not standardised the samples at all, but then it would be very hard to compare the data to the theoretical cdfs.

To produce `N` samples with arrival rate `rho` and plot a diagram, run
```
python fast_jm.py rho N
```

It saves the generated samples to a file in `data/`. It saves two other interesting thresholds which are computed as intermediate steps in the simulation: the time the last point arrives in a vacant region (saved to `rho1234-final-arrivals.csv`), and the first time after this final arrival at which there are no isolated balls (saved to `rho1234-iso-thresholds.csv`). It also saves the total number of points which arrive in an uncovered region.

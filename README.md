# johnson-mehl
Some experiments to test the predicted limiting CDFs for a Johnson-Mehl tessellation in a 2d box.

To produce `N` samples with arrival rate `rho` and plot a diagram, run
```
python fast_jm.py rho N
```

It saves the generated samples to a file in `data/`. It saves two other interesting thresholds: the time the last point arrives in a vacant region (saved to `rho1234-final-arrivals.csv`), and the first time after this final arrival at which there are no isolated balls (saved to `rho1234-iso-thresholds.csv`).

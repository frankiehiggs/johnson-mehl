import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import sys

from IPython.display import clear_output

from multiprocessing import Pool

from jmutils import *

def get_distance(a):
    rates = U**(-1/a)
    overtaken = get_overtake_times(rates,dists,fastest_index)
    assignments, times = assignment_function(seeds,rates,overtaken,resolution,max_time)
    last = np.unravel_index(np.argmax(times),times.shape)
    last_location = np.array( [last[i] / (resolution-1) for i in range(dimension)] )
    covered_by = assignments[last]
    return np.linalg.norm(last_location - seeds[covered_by])

if __name__=='__main__':
    # Reset the configuration
    n = 1000000
    dimension = 3
    resolution = 100
    exponents = np.linspace(0.5,3.5,31,endpoint=True)
    distances = np.zeros_like(exponents)
    count = 0 # To do: load count and distances from a saved file if it exists.
    if dimension == 2:
        assignment_function = assign_cells_random_radii
    elif dimension == 3:
        assignment_function = assign_cells_random_radii_3d
    elif dimension == 4:
        assignment_function = assign_cells_random_radii_4d
    else:
        raise Exception(f"We don't support d = {dimension} yet.")
    # "Meta-configuration" not affecting the simulation results.
    if len(sys.argv) >= 2:
        repetitions = int(sys.argv[1])
    else:
        repetitions = 10
    if len(sys.argv) >= 3:
        PARALLEL = int(sys.argv[2])
    else:
        PARALLEL = 1

    max_time = 2*( np.log(n) / (np.pi * n) )**(1/dimension)

    for k in trange(repetitions):
        seeds = sample_points(n,d=dimension)
        U = np.random.random(size=seeds.shape[0]) # Random heavy-tailed radii
        fastest_index = np.argmin(U)
        fastest_seed = seeds[fastest_index]
        dists = np.linalg.norm(seeds - fastest_seed, axis=1)
        with Pool(PARALLEL) as p:
            new_distance = p.map(get_distance, exponents)
        distances += new_distance
        count += 1 
        if k % 10 == 0:
            np.savez(f"data/n{n}d{dimension}.npz", distances=distances, count=count)

    fig, ax = plt.subplots()
    avg_distances = distances / count
    ax.plot(exponents,avg_distances)
    ax.set_title(f'n={n}, d={dimension}')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("average distance")
    plt.savefig(f'avg-distance-d{dimension}-n{n}.pdf')
    plt.close()
    
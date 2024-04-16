"""
Draws a picture of either a Voronoi tessellation or a Johnson-Mehl tessellation.
I may add other tessellations with random radii, that would be interesting
and not too difficult (perhaps something with E(Y^3) = \infty...)
"""

import numpy as np
from scipy.spatial import KDTree
from tqdm import trange
import networkx
import colorspace
from PIL import Image, ImageColor
import sys

from unconstrained import sample_points, prune_arrivals

def get_arrival_times( rho, max_time=1.0, R=0 ):
    N = np.random.poisson(lam=rho*max_time*(1+2*R)**2)
    return np.sort(np.random.uniform(low=0.0, high=max_time, size=N))
# Needs redefining because of my foolishly using a global variable (rng)
# to define the version of this function in unconstrained.py.

def get_ball_pixels(centre, radius, img_size):
    """
    Returns the indices of the pixels in the picture
    corresponding to a ball centred at a point in [0,1]^2
    of a given radius.
    Also saves the corresponding (squared) distances.
    """
    v = (img_size-1)*centre
    x,y = v[0], v[1]
    r = (img_size-1)*radius
    r2 = r*r
    min_i = max( 0, int(x-r) )
    max_i = min( img_size-1, int(x+r)+1 )
    min_j = max( 0, int(y-r) )
    max_j = min( img_size-1, int(y+r)+1 )
    in_ball = []
    sq_distances = []
    for i in range(min_i, max_i+1):
        for j in range(min_j, max_j+1):
            d2 = (x-i)**2 + (y-j)**2
            if d2 <= r2:
                in_ball.append((i,j))
                sq_distances.append(d2)
    return in_ball, sq_distances

def assign_cells( seeds, times, img_size, T=1.0 ):
    """
    Assigns all the pixels in an img_size x img_size picture
    to their respective Johnson-Mehl cells.
    T should be a decent upper bound on the coverage time - smaller T
    means we check fewer points.
    This is (a slightly simplified version of) Moulinec's algorithm.

    To do: add a "growth speeds" version for random radii.
    We don't have the compute a sqrt for that, so maybe I'll
    write it as a separate function.
    """
    min_cov_times = np.full((img_size,img_size),np.inf) # running minimum coverage times
    assignments = np.empty((img_size,img_size),dtype=int)

    for i in trange(len(times)):
        xi = seeds[i]
        ti = times[i]
        indices, d2s = get_ball_pixels(xi, T-ti, img_size)
        for k, ij_pair in enumerate(indices):
            cov_time = np.sqrt(d2s[k])/img_size + ti
            if cov_time < min_cov_times[ij_pair]:
                assignments[ij_pair] = i
                min_cov_times[ij_pair] = cov_time
    return assignments

def assign_cells_random_radii(seeds, rates, img_size, T=1.0):
    min_cov_times = np.full((img_size,img_size),np.inf) # running minimum coverage times
    assignments = np.empty((img_size,img_size),dtype=int)
    for i in trange(len(rates)):
        xi = seeds[i]
        gi = rates[i]
        gi2 = gi*gi
        indices, d2s = get_ball_pixels(xi, T*gi, img_size)
        for k, ij_pair in enumerate(indices):
            cov_time2 = d2s[k] / gi2
            if cov_time2 < min_cov_times[ij_pair]:
                assignments[ij_pair] = i
                min_cov_times[ij_pair] = cov_time2
    return assignments

def get_adjacency(cell_assignments):
    G = networkx.Graph()
    #G.add_nodes_from(range(cell_assignments.max()+1)) # Uncomment this to include cells with zero pixels
    N = cell_assignments.shape[0]
    for i in range(N-1): # All columns except the last
        for j in range(N-1): # All rows except the last
            G.add_edge(cell_assignments[i,j], cell_assignments[i+1,j])
            G.add_edge(cell_assignments[i,j], cell_assignments[i,j+1])
        G.add_edge(cell_assignments[i,N-1],cell_assignments[i+1,N-1])
    for j in range(N-1):
        G.add_edge(cell_assignments[N-1,j],cell_assignments[N-1,j+1])
    G.remove_edges_from(networkx.selfloop_edges(G)) # Not necessary for the colouring but if we want to look at the graph structure it makes it a bit cleaner.
    return G

def colour_graph(G):
    """
    Uses a greedy algorithm to colour G.
    The "colours" are just integers, which can be replaced
    with a suitable set of colours when drawing the picture later.
    Even with a few thousand cells I've never seen it use more than
    7 colours.

    Returns a dictionary indexed by the elements of G.nodes
    """
    cells = list(G.nodes).copy()
    colours = dict.fromkeys(G.nodes)
    
    np.random.shuffle(cells)
    for cell in cells:
        new_colour = 0
        while new_colour in [colours[v] for v in G.neighbors(cell)]:
            new_colour += 1
        colours[cell] = new_colour
    return colours

def jm_picture(rho, resolution):
    times = get_arrival_times(rho)
    seeds = sample_points(len(times))
    arrived = prune_arrivals(times, seeds)
    print(f'{len(arrived)} out of {len(times)} seeds germinated.')
    times = times[arrived]
    seeds = seeds[arrived]

    max_time = 2*( (2*np.log(rho) + 4*np.log(np.log(rho))) / (np.pi*rho) )**(1/3)
    I = assign_cells(seeds, times, resolution, T=max_time)

    cell_structure = get_adjacency(I)
    print(cell_structure)

    colours = colour_graph(cell_structure)
    print(f'We have a {max(colours.values())+1}-colouring of the cells.')

    c = colorspace.hcl_palettes().get_palette(name="Emrld")
    hex_colours = c(max(colours.values())+1)
    rgb_colours = [ImageColor.getcolor(col,"RGB") for col in hex_colours]

    data = np.empty((resolution, resolution, 3), dtype=np.uint8)
    N = I.shape[0]
    for i in range(N):
        for j in range(N):
            data[i,j,:] = rgb_colours[colours[I[i,j]]]

    image = Image.fromarray(data)
    image.show()

def random_radii_picture(n, resolution, distribution='uniform', p1=1.0, p2=1.0, p3=0.5):
    seeds = sample_points(n)

    if distribution == 'constant':
        second_moment = 1
        rates = np.ones(shape=n)
    elif distribution == 'uniform':
        second_moment = 1/3
        rates = np.random.uniform(size=n)
    elif distribution == 'exponential':
        second_moment = 1/(p1**2)
        rates = np.random.exponential(scale=p1, size=n)
    elif distribution == 'discrete':
        # Balls have, with equal probability, the two given radii
        second_moment = p3*p1*p1 + (1-p3)*p2*p2
        rates = p2 + (p1-p2)*np.random.binomial(1,p3,size=n)
    elif distribution == 'pareto':
        # Here we can have something which only just meets the moment conditions
        rates = p2*(np.random.pareto(p1,size=n)+1)
        if p1 <= 2:
            print("The first parameter of the Pareto distribution needs to be strictly greater than 2 so that second moments exist... we can still run the simulation, but it will be dominated by one or two massive components.")
            second_moment = 1
        else:
            second_moment = p1*p2**p1 / (p1 - 2)

    max_time = 2*np.sqrt( np.log(n) / (np.pi * n * second_moment) )
    I = assign_cells_random_radii(seeds, rates, resolution, T = max_time)

    cell_structure = get_adjacency(I)
    print(cell_structure)

    colours = colour_graph(cell_structure)
    print(f'We have a {max(colours.values())+1}-colouring of the cells.')

    c = colorspace.hcl_palettes().get_palette(name="SunsetDark")
    hex_colours = c(max(colours.values())+1)
    rgb_colours = [ImageColor.getcolor(col,"RGB") for col in hex_colours]

    data = np.empty((resolution, resolution, 3), dtype=np.uint8)
    N = I.shape[0]
    for i in range(N):
        for j in range(N):
            data[i,j,:] = rgb_colours[colours[I[i,j]]]

    image = Image.fromarray(data)
    image.show()

if __name__=='__main__':
    # rho = float(sys.argv[1])
    # resolution = int(sys.argv[2])
    # jm_picture(rho, resolution)
    if len(sys.argv) >= 5:
        random_radii_picture(int(sys.argv[1]), 1080, sys.argv[2], float(sys.argv[3]),float(sys.argv[4]))
    elif len(sys.argv) == 4:
        random_radii_picture(int(sys.argv[1]), 1080, sys.argv[2], float(sys.argv[3]))
    elif len(sys.argv) == 3:
        random_radii_picture(int(sys.argv[1]), 1080, sys.argv[2])
    else:
        random_radii_picture(int(sys.argv[1]), 1080)
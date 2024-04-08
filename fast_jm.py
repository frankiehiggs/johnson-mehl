"""
Finds the first time required for the Johnson-Mehl tesselation in 2d
to cover the unit square, when points are placed only inside the square.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
#from scipy.spatial import KDTree
from numba_kdtree import KDTree
import time
from tqdm import trange
import sys
from numba import jit

def save_data( filename, samples ):
    f = open(filename, 'a')
    for s in samples:
        f.write(str(s)+'\n')
    f.close()

@jit(nopython=True)
def sample_points( sample_size ):
    """
    Chooses points uniformly in [0,1]^2.
    """
    return np.random.random(size=(sample_size,2))

def get_arrival_times( rho, max_time=1.0 ):
    N = rng.poisson(lam=rho*max_time)
    return np.sort(rng.uniform(low=0.0, high=max_time, size=N))

@jit(nopython=True)
def prune_arrivals( times, locations ):
    """
    Given a list of arrival times and locations,
    returns the list of indices which are not covered by earlier points,
    i.e. those which arrived in empty space.
    
    A little slow, but saves a lot of time later.
    """
    N = len(times)
    indices = [0]
    for i in range(1,N):
        covered = False
        x = locations[i]
        t = times[i]
        for j in indices:
            vec = locations[j] - x
            radius = t - times[j]
            d2 = np.dot(vec,vec)
            if d2 < radius*radius:
                covered=True
                break
        if not covered:
            indices.append(i)
    return indices

@jit
def is_in_domain(x):
    if x[0] > 0 and x[0] < 1 and x[1] > 0 and x[1] < 1:
        return True
    else:
        return False

@jit(nopython=True)
def find_circle_corners(t, points_tree, times, d_matrix ):
    """
    Returns a list of "corners"--intersections between two circles.
    """
    corners = [] # This will be a list of numpy arrays, each of shape (1,2)
    N = len(times)
    radii = t - times # We'll only ever call the function with t > max(times).
    #candidate_pairs = points_tree.query_pairs(2*radii[0])
    #for i, j in candidate_pairs:
    for i in range(N):
        ri = radii[i]
        ri2 = ri*ri
        zi = points_tree.data[i]
        for j in range(i+1,N):
            rj = radii[j]
            # Since i<j, we have ri > rj.
            x = d_matrix[i,j]
            if x < ri+rj and x > ri-rj:
                x2 = x*x
                rj2 = rj*rj
                a = 0.5*(1 + (ri2 - rj2)/x2)
                b = np.sqrt( ri2/x2 - a*a )
                vect = points_tree.data[j] - zi
                perpx= vect[1]
                if perpx < 0:
                    perp = np.array( [perpx, -vect[0]] )
                    new_corner = zi + vect*a + perp*b
                else:
                    perp = np.array( [-perpx, vect[0]] )
                    new_corner = zi + vect*a + perp*b
                if is_in_domain(new_corner):
                    corners.append(new_corner)
    L = len(corners)
    output = np.empty(shape=(L,2))
    for i,corner in enumerate(corners):
        output[i] = corner
    return output

@jit(nopython=True)
def find_boundary_corners(t, points, times):
    """
    Returns a list of intersections between circles and the boundary.
    """
    radii = t - times
    N = len(times)
    corners = []
    for i in range(N):
        x,y = points[i]
        r = radii[i]
        rsq = r*r
        if x < r:
            delta = np.sqrt(rsq - x*x)
            if y+delta < 1:
                corners.append( np.array( (0, y+delta) ) )
            if y-delta > 0:
                corners.append( np.array( (0, y-delta) ) )
        if x > 1-r: # If r>0.5 both x<r and x>1-r are possible.
            delta = np.sqrt(rsq - (1-x)*(1-x) )
            if y+delta < 1:
                corners.append( np.array( (1, y+delta) ) )
            if y-delta > 0:
                corners.append( np.array( (1, y-delta) ) )
        if y < r:
            delta = np.sqrt(rsq - y*y)
            if x+delta < 1:
                corners.append( np.array( (x+delta, 0) ) )
            if x-delta > 0:
                corners.append( np.array( (x-delta, 0) ) )
        if y > 1-r:
            delta = np.sqrt(rsq - (1-y)*(1-y) )
            if x+delta < 1:
                corners.append( np.array( (x+delta, 0) ) )
            if x-delta > 0:
                corners.append( np.array( (x-delta, 0) ) )
    L = len(corners)
    output = np.empty(shape=(L,2))
    for i,corner in enumerate(corners):
        output[i] = corner
    return output

@jit(nopython=True)
def no_isolated_points_time(d_matrix, times):
    """
    Returns the first time after the arrival of the final point in a vacant space
    at which there are no isolated points.
    This is a function of the arrival times and the distances between points.
    """
    running_max = 2*times[-1]
    for i,ti in enumerate(times):
        running_min = 100 # Anything large.
        for j in range(len(times)):
            if j != i:
                running_min = min(d_matrix[i,j] + times[j], running_min)
        running_max = max(ti + running_min, running_max)
    return 0.5*running_max

@jit(nopython=True)
def is_covered(t, points_tree,times,d_matrix):
    # "points_tree" should be a KDTree.
    EPSILON = 0.0000001
    
    radii = t - times
    max_radius = radii[0]
    
    """
    Method 2: build a KDTree containing all the corners,
    then get all the coverage candidates at once.
    We can still terminate early if we find an uncovered point.
    """
    circle_corners = find_circle_corners(t,points_tree,times,d_matrix)
    corners = np.append(circle_corners,find_boundary_corners(t,points_tree.data,times),axis=0)
    all_covered = True
    if corners.size != 0:
        #corners_tree = KDTree(corners)
        #candidates = corners_tree.query_ball_tree(points_tree,max_radius)
        J = len(radii) # Number of points
        for corner in corners:
            covered = False
            for j in range(J):
                rj = radii[j] - EPSILON
                vect = points_tree.data[j] - corner
                d2 = np.dot(vect,vect)
                if d2 < rj*rj:
                    covered = True
                    break
            if not covered:
                all_covered = False
                break
    return all_covered

def coverage_time(rho, max_time=1.0, tolerance=0.00001, alpha=0.4):
    """
    Samples just the coverage time of the Johnson-Mehl process.
    """
    arrival_times = get_arrival_times(rho, max_time)
    locations = sample_points(len(arrival_times))
    # Prune the arrivals (slow but reduces the number of points drastically)
    indices = prune_arrivals(arrival_times, locations)
    arrival_times = arrival_times[indices]
    locations = locations[indices,:]
    
    d_matrix  = pdist(locations)
    #d2_matrix = euclidean_distances(locations,locations,squared=True)
    
    loc_tree = KDTree(locations)
    
    time_lb = no_isolated_points_time(d_matrix, arrival_times)
    time_ub = max_time
    alpha_prime = 1-alpha
    while time_ub - time_lb > tolerance:
        current_time = alpha_prime * time_lb + alpha * time_ub
        if is_covered(current_time, loc_tree,arrival_times,d_matrix):
            time_ub = current_time
        else:
            time_lb = current_time
    return 0.5*(time_lb + time_ub)

def three_times(rho,max_time=1.0,tolerance=0.000001,alpha=0.4,N_samples=1,file_prefix=None):
    """
    Very similar to coverage_time, but outputs the final arrival time,
    the isolation threshold, and the coverage time.
    """
    progress = trange(N_samples)
    final_arrivals = [None]*N_samples
    total_arrivals = [None]*N_samples
    iso_thresholds = [None]*N_samples
    coverage_times = [None]*N_samples
    for i in progress:
        progress.set_description('Generating arrival process')
        arrival_times = get_arrival_times(rho, max_time)
        locations = sample_points(len(arrival_times))
        # Prune the arrivals (slow but reduces the number of points drastically)
        progress.set_description('Pruning arrivals          ')
        indices = prune_arrivals(arrival_times, locations)
        arrival_times = arrival_times[indices]
        locations = locations[indices,:]
        
        final_arrival = arrival_times[-1]
        
        progress.set_description('Calculating pairwise dist ')
        d_matrix  = squareform(pdist(locations))
        #d2_matrix = euclidean_distances(locations,locations,squared=True)
        
        progress.set_description('Making k-dimensional tree ')
        loc_tree = KDTree(locations)

        progress.set_description('Connecting isolated points')
        iso_threshold = no_isolated_points_time(d_matrix, arrival_times)
        
        time_lb = iso_threshold
        time_ub = max_time
        alpha_prime = 1.0-alpha
        while time_ub - time_lb > tolerance:
            progress.set_description(f'Width {time_ub-time_lb:.1e}, aim {tolerance:.1e}')
            current_time = alpha_prime * time_lb + alpha * time_ub
            if is_covered(current_time, loc_tree,arrival_times,d_matrix):
                time_ub = current_time
            else:
                time_lb = current_time
        final_arrivals[i] = final_arrival
        total_arrivals[i] = len(indices)
        iso_thresholds[i] = iso_threshold
        coverage_times[i] = 0.5*(time_ub + time_lb)
    
    if file_prefix:
        save_data(f'{file_prefix}-rho{rho}-final-arrivals.csv',final_arrivals)
        save_data(f'{file_prefix}-rho{rho}-total-arrivals.csv',total_arrivals)
        save_data(f'{file_prefix}-rho{rho}-iso-thresholds.csv',iso_thresholds)
        save_data(f'{file_prefix}-rho{rho}-coverage-times.csv',coverage_times)
    else:
        save_data(f'data/rho{rho}-final-arrivals.csv',final_arrivals)
        save_data(f'data/rho{rho}-total-arrivals.csv',total_arrivals)
        save_data(f'data/rho{rho}-iso-thresholds.csv',iso_thresholds)
        save_data(f'data/rho{rho}-coverage-times.csv',coverage_times)

def limit(beta):
    return np.exp( - (4*np.pi)**(-1/3)*np.exp(-beta/3) - (2*np.pi*np.pi)**(-1/3)*4*np.exp(-beta/6) )
    
def chiu_limit(beta, rho):
    # Chiu used a different normalisation for the coverage time
    # in his weak law result (Thm 4).
    # So if we just use his limit formula naively
    # to predict the distribution using Penrose's
    # normalisation, then rho shows up in the "limit".
    # Asymptotically it shouldn't depend on rho though.
    c = np.log(rho*rho/np.pi)
    exponent = -0.25*(1/np.pi)*rho*rho*(c**(1/3))*(c+np.log(c))*np.exp(-(c**(2/3))*(beta + 2*np.log(rho) + 4*np.log(np.log(rho)))**(1/3))
    return np.exp(exponent)

def g(rho,t):
    return np.pi*rho*np.power(t,3) - 2*np.log(rho) - 4*np.log(np.log(rho))

def diagram_with_limit(rho, filename=None, outname=None):
    """
    Plots the limiting cdf and empirical cdf on the same axes. Requires the generated list of samples as input.
    """
    fig, ax = plt.subplots()
    if filename:
        #print("Loading samples...")
        samples = np.genfromtxt(filename)
    else:
        print("Please input the file name of some data.")
        return

    my_range = np.arange(min(-20,min(samples)-0.1),max(50,max(samples)+0.1),0.1)

    #print("Computing histogram...")
    normalised_samples = g(rho,samples)
    normalised_samples.sort()
    ax.plot(normalised_samples, (np.arange(samples.size)+1)/samples.size, 'b', linewidth=2, label="Empirical distribution")
    #print("Histogram computed!")
    
    p_limit = limit(my_range)
    ax.plot(my_range,p_limit,'k--',linewidth=1.5,label="Limiting cdf (from Penrose)")
    p_chiu = chiu_limit(my_range, rho)
    ax.plot(my_range,p_chiu,'r--',linewidth=1.5,label="Predicted cdf (from Chiu)")
    
    ax.set_ylim(0,1)
    ax.set_xlim(-10,40)
    ax.legend(loc='lower right')
    ax.set_title(f'Coverage time for a Johnson-Mehl process with rho={rho:.0e}')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    
    if outname:
        fig.savefig(outname)
        plt.close()
    else:
        plt.show()


if __name__=='__main__':
    rng = np.random.default_rng()
    try:
        rho = int(sys.argv[1]) # Arrival rate
        N = int(sys.argv[2])
    except:
        raise Exception("Two arguments required: the arrival rate rho, and the number of samples N")
    max_time = 2*( (2*np.log(rho) + 4*np.log(np.log(rho))) / (np.pi*rho) )**(1/3)
    print(f'\nrho = {rho}, upper bound {max_time:.3f}')
    
    three_times(rho,max_time=max_time, tolerance=0.0001,N_samples=N)
    diagram_with_limit(rho,filename=f'data/rho{rho}-coverage-times.csv',outname=f'diagrams/diagram-rho{rho}.png')



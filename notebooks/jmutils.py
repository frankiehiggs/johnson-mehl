# A collection of all the various tools I need to do experiments. Not involved in creating the Pareto video.
# Written for distance-experiment.ipynb, so may not be 100% compatible with the other notebooks yet.
import numpy as np
from numba import jit

@jit(nopython=True)
def sample_points( sample_size, R=0, d=2 ):
    """
    Chooses points uniformly in [-R,1+R]^2
    """
    return (1+2*R)*np.random.random(size=(sample_size,d)) - R*np.ones(d)

@jit(nopython=True)
def get_ball_pixels(centre, radius, img_size):
    """
    Returns the indices of the pixels in the picture
    corresponding to a ball centred at a point in [0,1]^d
    of a given radius.
    Also saves the corresponding (squared) distances.
    """
    in_ball = [(int(x),int(x)) for x in range(0)] # Funny expression creates a "typed list" for Numba. # I could also use indices of pixels (i.e. flattened coordinates)
    sq_distances = [np.float64(x) for x in range(0)]
    if radius > 0:
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
            dx2 = (x-i)*(x-i)
            if dx2 > r2:
                continue
            w = np.sqrt( r2 - dx2 )
            for j in range(max(int(y-w),min_j), min(int(y+w)+2,max_j+1)):
                d2 = dx2 + (y-j)**2
                if d2 <= r2:
                    in_ball.append((i,j))
                    sq_distances.append(d2)
    return in_ball, sq_distances

@jit(nopython=True)
def get_ball_pixels_3d(centre, radius, img_size):
    in_ball = [(int(x),int(x),int(x)) for x in range(0)]
    sq_distances = [np.float64(x) for x in range(0)]
    if radius > 0:
        v = (img_size-1)*centre
        x,y,z = v[0], v[1], v[2]
        r = (img_size-1)*radius
        r2 = r*r
        min_k = max(0, int(z-r))
        max_k = min( img_size-1, int(z+r)+1 )
        for k in range(min_k, max_k+1):
            dz2 = (z-k)**2
            projected_radius = np.sqrt( r2 - dz2 ) / (img_size - 1)
            projected_disc,projection_sq_distances = get_ball_pixels(centre[0:2],projected_radius,img_size)
            for n,ij in enumerate(projected_disc):
                in_ball.append( (ij[0],ij[1],k) )
                sq_distances.append( dz2 + projection_sq_distances[n] )
    return in_ball, sq_distances

@jit(nopython=True)
def assign_cells_random_radii(seeds, rates, overtaken, img_size, T=1.0):
    min_cov_times = np.full((img_size,img_size),np.inf) # running minimum coverage times
    assignments = np.full((img_size,img_size),-1,dtype=np.int64)
    attempts = 0
    while -1 in assignments:
        # if attempts > 0:
            # print(f'Attempt {attempts}')
        for i in range(len(rates)):
            if attempts > 0 and overtaken[i] < 0.5*T: # i.e. if there are no new pixels to check in this ball
                continue
            xi = seeds[i]
            gi = rates[i]
            gi2 = gi*gi
            indices, d2s = get_ball_pixels(xi, gi*min(T,overtaken[i]), img_size)
            for k, ij_pair in enumerate(indices):
                cov_time2 = d2s[k] / gi2
                if cov_time2 < min_cov_times[ij_pair]:
                    assignments[ij_pair] = i
                    min_cov_times[ij_pair] = cov_time2
        T *= 2
        attempts += 1
    return assignments, min_cov_times

@jit(nopython=True)
def assign_cells_random_radii_3d(seeds, rates, overtaken, img_size, T=1.0):
    min_cov_times = np.full((img_size,img_size,img_size),np.inf) # running minimum coverage times
    assignments = np.full((img_size,img_size,img_size),-1,dtype=np.int64)
    attempts = 0
    while -1 in assignments:
        # if attempts > 0:
            # print(f'Attempt {attempts}')
        for i in range(len(rates)):
            if attempts > 0 and overtaken[i] < 0.5*T: # i.e. if there are no new pixels to check in this ball
                continue
            xi = seeds[i]
            gi = rates[i]
            gi2 = gi*gi
            indices, d2s = get_ball_pixels_3d(xi, gi*min(T,overtaken[i]), img_size)
            for n, ijk in enumerate(indices):
                cov_time2 = d2s[n] / gi2
                if cov_time2 < min_cov_times[ijk]:
                    assignments[ijk] = i
                    min_cov_times[ijk] = cov_time2
        T *= 2
        attempts += 1
    return assignments, min_cov_times

@jit(nopython=True)
def get_overtake_times(rates,dists,fastest_index):
    overtaken = np.empty(len(rates),dtype=np.float64)
    fastest_rate = rates[fastest_index]
    for i,speed in enumerate(rates):
        if i == fastest_index:
            overtaken[i] = np.inf
            continue
        else:
            overtaken[i] = dists[i] / (fastest_rate - speed)
    return overtaken

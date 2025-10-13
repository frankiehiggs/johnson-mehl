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
    if radius == 0:
        pass # Leave the arrays empty
    elif radius > np.sqrt(2): # Everything is covered
        # To do: there is surely some faster way of doing this.
        # Maybe do the check in assign_cells and then I can
        # avoid saving this enormous list of indices.
        # I don't really see the advantage of computing the distances
        # inside this function.
        v = (img_size-1)*centre
        x,y = v[0], v[1]
        for i in range(img_size):
            for j in range(img_size):
                in_ball.append((i,j))
                sq_distances.append( (x-i)**2 + (y-j)**2 )
    else:
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
    """
    To do: Idea for improving this function.
    Follow the example of the 4d one:
    first sample all possible i,j pairs for
    the first two coordinates using
    get_ball_pixels,
    then the possible k values are easy to compute
    for each pair (i,j).
    """
    in_ball = [(int(x),int(x),int(x)) for x in range(0)]
    sq_distances = [np.float64(x) for x in range(0)]
    if radius == 0:
        pass # Leave the arrays empty
    elif radius > np.sqrt(3): # Everything is covered
        v = (img_size-1)*centre
        x,y,z = v[0], v[1], v[2]
        for i in range(img_size):
            for j in range(img_size):
                for k in range(img_size):
                    in_ball.append((i,j,k))
                    sq_distances.append( (x-i)**2 + (y-j)**2 + (z-k)**2 )
    else:
        x = centre[0]*(img_size-1)
        r = (img_size-1)*radius
        r2 = r*r
        min_i = max(0, int(x-r))
        max_i = min( img_size-1, int(x+r)+1 )
        for i in range(min_i, max_i+1):
            dx2 = (x-i)**2
            projected_radius = np.sqrt( r2 - dx2 ) / (img_size - 1)
            projected_disc,projection_sq_distances = get_ball_pixels(centre[1:3],projected_radius,img_size)
            for n,jk in enumerate(projected_disc):
                in_ball.append( (i,jk[0],jk[1]) )
                sq_distances.append( dx2 + projection_sq_distances[n] )
    return in_ball, sq_distances

@jit(nopython=True)
def get_ball_pixels_4d(centre, radius, img_size):
    """
    Computes an array of 2d balls along 2 axes.
    """
    in_ball = [(int(x),int(x),int(x),int(x)) for x in range(0)]
    sq_distances = [np.float64(x) for x in range(0)]
    if radius == 0:
        pass # Leave the arrays empty
    elif radius > 2: # Everything is covered
        v = (img_size-1)*centre
        w,x,y,z = v[0], v[1], v[2], v[3]
        for i in range(img_size):
            for j in range(img_size):
                for k in range(img_size):
                    for l in range(img_size):
                        in_ball.append((i,j,k,l))
                        sq_distances.append( (w-i)**2 + (x-j)**2 + (y-k)**2 + (z-l)**2 )
    else:
        xys, xy_d2s = get_ball_pixels(centre[0:2],radius,img_size)
        r = (img_size-1)*radius
        r2 = r*r
        for n, xy_d2 in enumerate(xy_d2s):
            i,j = xys[n]
            projected_radius = np.sqrt( r2 - xy_d2 ) / (img_size - 1)
            pqs, pq_sq_dists = get_ball_pixels(np.array(xys[n])/(img_size-1),projected_radius,img_size)
            for m, pq_d2 in enumerate(pq_sq_dists):
                in_ball.append( (i,j,pqs[m][0],pqs[m][1]) )
                sq_distances.append( xy_d2 + pq_d2 )
    return in_ball, sq_distances

# To do:
# A decent Python programmer wouldn't just return
# a bit list of tuples with indices (taking up lots of
# memory when img_size or the dimension is high).
# They'd return an iterator / generator
# / something like that.
# I don't know whether that'd be compatible with Numba,
# but it might be pretty efficient.
# Now all I need to do is find a decent programmer...

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
def assign_cells_random_radii_4d(seeds, rates, overtaken, img_size, T=1.0):
    min_cov_times = np.full((img_size,img_size,img_size,img_size),np.inf) # running minimum coverage times
    assignments = np.full((img_size,img_size,img_size,img_size),-1,dtype=np.int64)
    attempts = 0
    while -1 in assignments:
        if attempts > 0:
            print(f'Attempt {attempts}')
        for i in range(len(rates)):
            if attempts > 0 and overtaken[i] < 0.5*T: # i.e. if there are no new pixels to check in this ball
                continue
            xi = seeds[i]
            gi = rates[i]
            gi2 = gi*gi
            indices, d2s = get_ball_pixels_4d(xi, gi*min(T,overtaken[i]), img_size)
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

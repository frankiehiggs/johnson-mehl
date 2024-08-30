import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm, trange
import networkx
import colorspace
from PIL import Image, ImageColor, ImageDraw
import sys
from pylatex import Document, Command, Math, NoEscape, Package
from pdf2image import convert_from_path
from numba import jit
import os
import gc

from unconstrained import sample_points
from draw_jm import get_adjacency, colour_graph, get_ball_pixels#, assign_cells_random_radii

@jit(nopython=True)
def assign_cells_random_radii(seeds, rates, overtaken, img_size, T=1.0):
    """
    Not-so-optimal algorithm: if there are uncovered points it doubles the final time
    and runs everything again.
    Moulinec's algorithm will probably be quite a lot faster,
    and I should implement it if I'm going to make this code available.
    """
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
    return assignments

# @jit(nopython=True)
# def assign_cells_random_radii(seeds, rates, img_size, T=1.0):
#     """
#     This is more-or-less Moulinec's algorithm.
#     It might be faster than the approach above if T is tuned correctly.
#     But if T isn't tuned carefully, it might be much slower.
#     """
#     min_cov_times = np.full((img_size,img_size),np.inf) # running minimum coverage times
#     assignments = np.full((img_size,img_size),-1,dtype=np.int64)
#     for i in range(len(rates)):
#         xi = seeds[i]
#         gi = rates[i]
#         gi2 = gi*gi
#         indices, d2s = get_ball_pixels(xi, T*gi, img_size)
#         for k, ij_pair in enumerate(indices):
#             cov_time2 = d2s[k] / gi2
#             if cov_time2 < min_cov_times[ij_pair]:
#                 assignments[ij_pair] = i
#                 min_cov_times[ij_pair] = cov_time2
#         for x,y in zip(*np.where(assignments==-1)):
#             pos = np.array([x,y])
#             for i in range(len(rates)):
#                 xi = seeds[i]
#                 gi = rates[i]
#                 gi2 = gi*gi
#                 d2 = np.linalg.norm( pos - xi*(img_size-1) )
#                 cov_time2 = d2 / gi2
#                 if cov_time2 < min_cov_times[x,y]:
#                     assignments[x,y] = i
#                     min_cov_times[x,y] = cov_time2
#     return assignments

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

def create_latex(a,outname,dpi_factor=1.35):
    if os.path.isfile(f'latex/{outname}.png'):
        return
    else:
        geometry_options = {"rmargin":"11cm"}
        doc = Document('basic',geometry_options=geometry_options)
        doc.packages.append(Package('amssymb'))
        if a <= 2:
            doc.append(NoEscape(f'Currently $a = {a:.2f}$,' + ' so $$\mathbb{E}[ Y_i^{' + f'{a:.2f}' + ' - \\varepsilon} ] < \\infty,$$ but $$\mathbb{E}[ Y_i^{' + f'{a:.2f}' + '}] = \\infty.$$'))
        else:
            doc.append(NoEscape(f'Currently $a = {a:.2f}$,' + ' so $$\mathbb{E}[ Y_i^{' + f'{2}' + ' + \\varepsilon} ] < \\infty,$$ but $$\mathbb{E}[ Y_i^{' + f'{a:.2f}' + '}] = \\infty.$$'))
        doc.generate_pdf(f'latex/temppdf{a:.2f}',clean_tex=True)
        x1, y1, x2, y2 = 340, 350, 1140, 800
        conv = convert_from_path(f'latex/temppdf{a:.2f}.pdf',dpi=200*dpi_factor)[0]
        # Does the line above cause a memory leak? Did it when I didn't have [0] at the end?
        # I could just create the files in advance, I suppose. Maybe in a separate script.
        im = conv.crop([x1,y1,x2,y2])
        im.save(f'latex/{outname}.png')
        im.close()
        os.remove(f'latex/temppdf{a:.2f}.pdf')
        return

def add_frame_metadata(im, a, m=1.0, panel_width = 840, font_size=20,dpi_factor=1.35):
    """
    Given an NxN image representing the tessellation where each random radius
    is Pareto distributed with scale m and shape a,
    extends the right-hand side of the image by adding a panel containing
    the values of a and m,
    and saying something about the moments.
    """
    height = im.height
    width = im.width
    expanded_im = Image.new(im.mode, (width+panel_width,height), (255,255,255))
    expanded_im.paste(im,(0,0))
    # Now let's write the metadata
    draw = ImageDraw.Draw(expanded_im)
    draw.text((width+0.05*panel_width,0.025*height),"Model parameters",fill="black",font_size=2.5*font_size)
    # draw.text((width+0.05*panel_width,0.15*height),"Each ball has growth rate",font_size=font_size,fill="black")
    with Image.open('latex/yi-def.PNG') as yi_def:
        expanded_im.paste(yi_def,(int(width+0.03*panel_width),int(0.12*height)))
    create_latex(a,f'temp{a:.3f}',dpi_factor) # If the file exists it loads it, otherwise it makes it.
    with Image.open(f'latex/temp{a:.3f}.png') as moments:
        expanded_im.paste(moments,(int(width+0.03*panel_width),int(0.35*height)))
    # draw.text((width+0.5*panel_width,0.2*height),f'a = {a:.4f}', fill="black",font_size=font_size)
    # os.remove(f'latex/temp{a:.3f}.png')
    return expanded_im

# def getassignments(a,seeds,U):
#     n = 100
#     resolution = 1080
#     max_time = 2*np.sqrt( np.log(n) / (np.pi * n) )
#     if a>2:
#         m = ((a-2.0)/a)**(1/a)
#     else:
#         m = 1.0
#     rates = m*U**(-a)
#     assignments = assign_cells_random_radii(seeds, rates, resolution, T=max_time)
#     print(f'a = {a:.3f} finished!')
#     return assignments

if __name__=='__main__':
    # multiprocessing.set_start_method('spawn')
    if len(sys.argv) < 3:
        raise Exception("Please give both the arguments: n, resolution")
    fileprefix = "frames/pareto-"
    n = int(sys.argv[1])
    resolution = int(sys.argv[2])
    nframes = 276
    exponents = np.linspace(3.00,0.25,num=nframes,endpoint=True)

    max_time = 2*np.sqrt( np.log(n) / (np.pi * n) )
    
    # np.random.seed(RANDOMSEED)
    print("Sampling arrival locations")
    n = np.random.poisson(lam=n) # This line turns a binomial point process into a Poisson point process.
    seeds = sample_points(n)
    U = np.random.random(size=seeds.shape[0])
    
    # Idea for speeding things up:
    # All the balls apart from the fastest one are overtaken at some time (it could be after the coverage time) by the fastest-growing ball.
    # The time each ball is overtaken is simply d/(G_max - G), where G is that ball's rate and d is the distance between it and the fastest-growing ball.
    # Therefore we can shrink most of the balls substantially and save a lot of time.
    # (since most rates G will be a lot smaller than G_max, we divide the radii by something which is almost G_max).
    # We only need to compute the distances once, since the fastest ball is always the fastest,
    # but we do need to recompute the difference in rates for every exponent.
    fastest_index = np.argmin(U)
    fastest_seed = seeds[fastest_index]
    dists = np.linalg.norm(seeds - fastest_seed, axis=1)

    supG = networkx.Graph()
    for a in tqdm(exponents):
        rates = U**(-1/a)
        overtaken = get_overtake_times(rates,dists,fastest_index)
        assignments = assign_cells_random_radii(seeds, rates, overtaken, resolution, T=max_time)
        supG.update(get_adjacency(assignments)) # Add new edges to supG
    
    print(supG)
    print("We have the adjacency graphs for each frame. Now colouring the 'all-time adjacency graph'...")
    
    # Export colours, assignments etc.
    np.savez('samples.npz', seeds=seeds, U=U, dists=dists)
    networkx.write_adjlist(supG, 'supG.adjlist') # We can send this to a Sagemath script if we want an optimal colouring (and are very patient)

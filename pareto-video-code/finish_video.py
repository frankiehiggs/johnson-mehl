import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map
import networkx
import colorspace
from PIL import Image, ImageColor, ImageDraw
import sys
from pylatex import Document, Command, Math, NoEscape, Package
from pdf2image import convert_from_path
from multiprocessing import Pool
from numba import jit
import os
import json

from unconstrained import sample_points
from draw_jm import get_adjacency, colour_graph, get_ball_pixels#, assign_cells_random_radii

from compute_adjacency import *

if __name__=='__main__':
    if len(sys.argv) < 4:
        raise Exception("Please give all the arguments: n, resolution, PARALLEL")
    fileprefix = "frames/pareto-"
    n = int(sys.argv[1]) # 20000
    resolution = int(sys.argv[2])
    nframes = 276
    exponents = np.linspace(3.00,0.25,num=nframes,endpoint=True)
    PARALLEL = int(sys.argv[3])
    
    # Load in the seeds, locations and colours
    with np.load('samples.npz') as data:
        seeds = data['seeds']
        U = data['U']
    def keystoint(x):
        return {int(k):v for k, v in x} # Convert keys from str to int.
    with open('colouring.json','r') as colfile:
        colours = json.load(colfile,object_pairs_hook=keystoint)
    
    # Import colours, assignments etc.
    print(f'We have a {max(colours.values())+1}-colouring of the cells.')
    c = colorspace.hcl_palettes().get_palette(name="SunsetDark")
    hex_colours = c(max(colours.values())+1)
    rgb_colours = [ImageColor.getcolor(col,"RGB") for col in hex_colours]
    print("Drawing the frames.")

    max_time = 2*np.sqrt( np.log(n) / (np.pi * n) )

    def makeframe(i):
        a = exponents[i]
        if a>2:
            m = ((a-2.0)/a)**(1/a)
        else:
            m = 1.0
        rates = m*U**(-1/a)
        I = assign_cells_random_radii(seeds,rates,resolution,T=max_time)
        data = np.empty((resolution, resolution, 3), dtype=np.uint8)
        for x in range(resolution):
            for y in range(resolution):
                data[x,y,:] = rgb_colours[colours[I[x,y]]]
        with Image.fromarray(data) as image:
            image = add_frame_metadata(image,a)
            # image.show()
            image.save(fileprefix+f'{str(i).zfill(6)}.png')
        return

    process_map(makeframe, range(len(exponents)), max_workers=PARALLEL, leave=True)
    
    # for i in trange(len(exponents),leave=False):
    #     a = exponents[i]
    #     I = assignments[i]
    #     data = np.empty((resolution, resolution, 3), dtype=np.uint8)
    #     for x in range(resolution):
    #         for y in range(resolution):
    #             data[x,y,:] = rgb_colours[colours[I[x,y]]]
    #     image = Image.fromarray(data)
    #     image = add_frame_metadata(image,a)
    #     # image.show()
    #     image.save(fileprefix+f'{a:.5f}.png')
    print("Done! Go and find your frames in the frames/ folder")

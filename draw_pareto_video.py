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

from unconstrained import sample_points
from draw_jm import get_adjacency, colour_graph, get_ball_pixels#, assign_cells_random_radii

@jit(nopython=True)
def assign_cells_random_radii(seeds, rates, img_size, T=1.0):
    min_cov_times = np.full((img_size,img_size),np.inf) # running minimum coverage times
    assignments = np.full((img_size,img_size),-1,dtype=np.int64)
    while -1 in assignments:
        for i in range(len(rates)):
            xi = seeds[i]
            gi = rates[i]
            gi2 = gi*gi
            indices, d2s = get_ball_pixels(xi, T*gi, img_size)
            for k, ij_pair in enumerate(indices):
                cov_time2 = d2s[k] / gi2
                if cov_time2 < min_cov_times[ij_pair]:
                    assignments[ij_pair] = i
                    min_cov_times[ij_pair] = cov_time2
        T *= 2
    return assignments

def create_latex(a,outname,dpi_factor=1.35):
    if os.path.isfile(f'latex/{outname}.png'):
        return
    else:
        geometry_options = {"rmargin":"11cm"}
        doc = Document('basic',geometry_options=geometry_options)
        doc.packages.append(Package('amssymb'))
        if a <= 2:
            doc.append(NoEscape(f'Currently $a = {a:.2f}$,' + ' so $$\mathbb{E}[ Y_i^{' + f'{a:.2f}' + ' - \\varepsilon} ] < \\infty,$$ but $$\mathbb{E}[ Y_i^{' + f'{a:.2f}' + '}] = \\infty.$$'))
            doc.append(NoEscape("Since we have are no second moments, we let $m = 1$."))
        else:
            doc.append(NoEscape(f'Currently $a = {a:.2f}$,' + ' so $$\mathbb{E}[ Y_i^{' + f'{2}' + ' + \\varepsilon} ] < \\infty,$$ but $$\mathbb{E}[ Y_i^{' + f'{a:.2f}' + '}] = \\infty.$$'))
            doc.append(NoEscape("The radii have second moments, so we choose $m = m(a)$ so that $\mathbb{E}[Y_i^2] = 1$."))
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
    create_latex(a,f'temp{a:.3f}',dpi_factor) # Make this conditional on the file not already existing.
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
    os.system("rm -f frames/pareto-*")
    fileprefix = "frames/pareto-"
    n = 100 #20000
    resolution = 1080
    nframes = 276
    exponents = np.linspace(3.00,0.25,num=nframes,endpoint=True)
    RANDOMSEED = 20240422 # Fixing a seed doesn't seem to work - is a different generator sneaking in somewhere?
    # If I remove the randomness from colour_graph() then the whole thing is a function of `seeds` and `U`: I could save these.
    PARALLEL = 8

    max_time = 2*np.sqrt( np.log(n) / (np.pi * n) )
    
    np.random.seed(RANDOMSEED)
    print("Sampling arrival locations")
    seeds = sample_points(n)
    U = np.random.random(size=seeds.shape[0])

    def getgraphs(a):
        if a>2:
            m = ((a-2.0)/a)**(1/a)
        else:
            m = 1.0
        rates = m*U**(-1/a) # When a is small (less than 0.2, say) this is pretty unstable.
        assignments = assign_cells_random_radii(seeds, rates, resolution, T=max_time)
        # print(f'a = {a:.3f} assigned!')
        graph = get_adjacency(assignments)
        return graph

    # Problem: if we're making a lot of frames, then we run out of memory if we store all the assignments.
    # Here's my (slow but memory-saving) idea:
    # 1. Compute the adjacency graphs (involving finding the assignments for a given exponent, making the graph, then throwing away the assignments). We can store all the graphs.
    # 2. Then take the supremum graph to get the colouring of cells.
    # 3. Then re-compute all the assignments to draw cells - this way we don't need to store assignments.
    #
    # If storing all the graphs still takes up too much memory, then we can just keep a "running supremum" graph.
    # That's harder to parallelise, but I think it can be done using a multiprocessing.Manager

    print("Calculating adjacency structure of the cells (this is the slowest step)...")
    ## Parallel processing
    graphs = process_map(getgraphs,exponents,max_workers=PARALLEL, leave=False)
    supG = graphs[0]
    for i in range(1,len(graphs)):
        supG.update(graphs[i])
    print(supG)
    
    ## Not parallel, but with a running-maximum graph.
    # supG = getgraphs(exponents[0])
    # for i in trange(1,len(exponents),leave=False):
        # supG.update(getgraphs(exponents[i]))
    # print(supG)
    
    print("We have the adjacency graphs for each frame. Now colouring the 'all-time adjacency graph'...")
    
    colours = colour_graph(supG)
    print(f'We have a {max(colours.values())+1}-colouring of the cells.')
    c = colorspace.hcl_palettes().get_palette(name="SunsetDark")
    hex_colours = c(max(colours.values())+1)
    rgb_colours = [ImageColor.getcolor(col,"RGB") for col in hex_colours]
    print("Drawing the frames.")

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

    process_map(makeframe, range(len(exponents)), max_workers=PARALLEL, leave=False)
    
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

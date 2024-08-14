import json
import networkx as nx
from sage.all import *
from sage.graphs.graph_coloring import vertex_coloring

if __name__=='__main__':
    supG = nx.read_adjlist('supG.adjlist')
    G = Graph(supG)
    # colour_classes = vertex_coloring(G,k=None,value_only=False) # Returns an optimal colouring. Can be slow if the graph has many vertices.
    ncolours = 10 # Things are faster if we allow non-optimal colourings.
    colour_classes = vertex_coloring(G, k=ncolours)
    while not colour_classes:
        ncolours += 1
        colour_classes = vertex_coloring(G, k=ncolours)
    colours = {}
    for i,col_class in enumerate(colour_classes):
        for v in col_class:
            colours[v] = i
    
    with open('colouring.json','w') as colfile:
        json.dump(colours,colfile)
        
    # Unless we really want an optimal colouring,
    # I may just replace all this with a greedy algorithm.

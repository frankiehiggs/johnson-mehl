#!/bin/bash

# Config. Only these three things should change (and I don't really plan to change the resolution):
n=100
resolution=1080
PARALLEL=8

# Doing stuff
rm -f frames/pareto-*
python3 compute_adjacency.py $n $resolution $PARALLEL
sage computer_colouring.py
python3 finish_video.py $n $resolution $PARALLEL

echo "Generating video now"

ffmpeg -framerate 60 -pattern_type glob -i 'frames/pareto-*.png' -c:v libx264 PARETO-VIDEO.mp4

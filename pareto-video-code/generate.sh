#!/bin/bash
set -e # Stop the whole script if anything goes wrong.

# Config. Only these three things should change (and I don't really plan to change the resolution):
n="$1"
resolution=1080
PARALLEL=8

# Doing stuff
rm -f frames/pareto-*
python3 compute_adjacency.py $n $resolution
#sage compute_colouring.py # If we've got it set up to find an optimal colouring.
python3 finish_video.py $n $resolution $PARALLEL

echo "Attempting to generate video"

datestamp=$(date "+%Y%m%d")
timestamp=$(date "+%H%M")
ffmpeg -framerate 15 -loglevel 8 -pattern_type glob -i 'frames/pareto-*.png' -c:v libx264 "PARETO-VIDEO-n$((n))-$((datestamp))-$((timestamp)).mp4"

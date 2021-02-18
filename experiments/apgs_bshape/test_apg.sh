#!/bin/bash
for i in 0 1 4 9
do
python rws_apg.py --timesteps=20 --num_objects=3 --num_sweeps=$i --device=cuda:0
done
#!/bin/bash
for i in 0 1 4 9
do
python rws_apg.py --timesteps=20 --num_objects=4 --num_sweeps=$i --device=cuda:1
done
#!/bin/bash

#SBATCH --job-name=timesegment
#SBATCH --time=100:00:00
#SBATCH --output=logs/identifiability-%j.out
#SBATCH --error=logs/identifiability-%j.err
#SBATCH -n 10
#SBATCH -N 1-1

ipython timesegment_matching.py $i $j

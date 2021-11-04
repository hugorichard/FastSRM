#!/bin/bash

#SBATCH --job-name=timesegment
#SBATCH --time=100:00:00
#SBATCH --output=logs/timesegment-%j.out
#SBATCH --error=logs/timesegment-%j.err
#SBATCH --cpus-per-task=10

ipython timesegment_matching.py $i $j

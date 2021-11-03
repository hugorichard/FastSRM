#!/bin/bash

#SBATCH --job-name=timesegment
#SBATCH --output=logs/timesegment_%i_%j.out
#SBATCH --time=100:00:00
#SBATCH -n 20
#SBATCH -N 1-1

ipython timesegment_matching.py $i $j

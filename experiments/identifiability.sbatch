#!/bin/bash

#SBATCH --job-name=identifiability
#SBATCH --time=100:00:00
#SBATCH --output=logs/identifiability-%j.out
#SBATCH --error=logs/identifiability-%j.err
#SBATCH -c 10
#SBATCH -N 1-1

ipython identifiability.py $i

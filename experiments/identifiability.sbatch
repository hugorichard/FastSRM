#!/bin/bash

#SBATCH --job-name=identifiability
#SBATCH --time=100:00:00
#SBATCH --output=logs/identifiability-%j.out
#SBATCH --error=logs/identifiability-%j.err
#SBATCH --cpus-per-task=10

ipython identifiability.py $i

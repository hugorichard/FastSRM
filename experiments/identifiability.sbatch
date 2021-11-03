#!/bin/bash

#SBATCH --job-name=identifiability
#SBATCH --output=logs/identifiability_%i_%j.out
#SBATCH --time=100:00:00
#SBATCH -n 10
#SBATCH -N 1-1

ipython identifiability.py $i
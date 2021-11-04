#!/bin/bash

for i in 'brainiak' 'fast'
do
    export i;
    sbatch identifiability.sbatch -o logs/identifiability_$i.out
done

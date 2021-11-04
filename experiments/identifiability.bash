#!/bin/bash

for i in 'brainiak' 'fast'
do
    export i;
    sbatch identifiability.sbatch --output=logs/identifiability_$i.out ;
done

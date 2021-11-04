#!/bin/bash

a = 'identifiability'
for i in 'brainiak' 'fast'
do
    sbatch --output=$a.$i.out --export=i=$i identifiability.sbatch
done

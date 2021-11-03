#!/bin/bash

for i in 'brainiak' 'fastsrm'
do
    export i;
    sbatch identifiability.sbatch
done

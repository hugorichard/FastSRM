#!/bin/bash

for component in '5' '10' '20' '50'
do
    for dataset in 'gallant' 'forrest' 'raiders' 'sherlock'
    do
        ipython timesegment_matching_memory.py $dataset $component &
    done
    wait
done

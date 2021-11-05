#!/bin/bash

for i in '5' '10' '20' '50'
do
    for j in 'det' 'prob' 'fastdet' 'fastprob'
    do
        ipython timesegment_matching.py $i $j
    done
done

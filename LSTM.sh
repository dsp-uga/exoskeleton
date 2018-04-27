#!/bin/bash

for i1 in .8 .6 .4 .2
do
    for i2 in .7 .5 .3 .1
    do
	for j1 in .8 .6 .4 .2
	do
	    for j2 in .7 .5 .3 .1
	    do
		for k in 1 2 3 4 5
		do
		    python LSTM.py -n 512 512 -d $i1 $i2 -v relu relu -r $j1 $j2 -f Data/NewClean/* -g -b 500 -R $k
		done
	    done
	done
    done
done

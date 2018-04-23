x#!/bin/bash

for s in 128 512 1024
do
  for i in .2 .15 .1
  do
      for j in .8 .7 .6 .5 .4 .3 .2 .1
      do
  	for k in 1 2 3 4 5
  	do
  	    python LSTM.py -n $s $s -d $i $i -v relu relu -r $j $j -f Data/Clean/* -e 50 -g -b 100 -R $k
  	done
      done
  done
  for j in .4 .3 .2 .1
  do
      for k in 1 2 3 4 5
      do
  	     python LSTM.py -n $s $s -d $i $i -v relu relu -r $j $j -f Data/Clean/* -e 50 -g -b 100 -R $k
      done
  done
done

for s in 128 512 1024
do
  for i in .2 .15 .1
  do
      for j in .8 .7 .6 .5 .4 .3 .2 .1
      do
  	for k in 1 2 3 4 5
  	do
  	    python LSTM.py -n $s $s $s -d $i $i $i -v relu relu relu -r $j $j $j -f Data/Clean/* -e 50 -g -b 100 -R $k
  	done
      done
  done
  for j in .4 .3 .2 .1
  do
      for k in 1 2 3 4 5
      do
  	     python LSTM.py -n $s $s $s -d $i $i $i -v relu relu relu -r $j $j $j -f Data/Clean/* -e 50 -g -b 100 -R $k
      done
  done
done

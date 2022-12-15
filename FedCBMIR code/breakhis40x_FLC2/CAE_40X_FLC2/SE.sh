#!/bin/bash 

for ((i = 0; i < 4; i++)) 
	do echo ${is[i]} ${js[i]}
done

for i j in {1 2 3 4 5 6 7 8} {03 06 03 06 05 10 05 10}
do
    echo exp_000$i, round_00$j
done 

for j in 03 06 03 06 05 10 05 10
do 
    echo round_00$j
done
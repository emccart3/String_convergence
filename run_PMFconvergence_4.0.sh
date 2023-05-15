#!/bin/bash

maxit=25

top=`pwd`

for i in $(seq 1 1 ${maxit}); do
    it=$(printf "%02d" ${i})
    dir=it$(printf "%02d" ${i})
    if [ ! -f string.${it}.dat ]; then
        cd ${top}/${dir}
        for j in img*disang; do
            grep r2 ${j} | head -n2 | awk '{print $6}' | head -n1 >> RC1.dat
            grep r2 ${j} | head -n2 | awk '{print $6}' | tail -n1 >> RC2.dat
        done
        paste RC1.dat RC2.dat > string.${it}.dat
        cp string.${it}.dat ${top}
        rm RC1.dat RC2.dat
        cd ${top}
    fi
done


for i in $(seq -w 1 ${maxit}); do
    cd it${i}
    if [ ! -f path.pkl ]; then 
        echo it${i} is missing path.pkl... you donut
    else
		if [ ! -f analysis/p.dat ]; then 
        	ndfes-PlotMultistringPath.py -P path.pkl -c analysis/metafile.all.chk -o analysis/p.dat
		fi
    fi
    cd ${top}
done

./PMFconvergence_4.0.py -i ${maxit} -d p.dat -s # use -s to analyze projected string and -p to analyze pmf. -r is no longer necessary

#!/bin/bash
INPUT=../studies_info.csv
CORES=`cat /proc/cpuinfo | grep processor | wc -l`
JOBS=`ps -df | grep python | wc -l`
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read phs BDC_study_name official_study_name short_name harmonized phs_list ID_varName
do
	while [ $[$CORES + 1] -le $JOBS ]; do
		sleep 2
		JOBS=`ps -df | grep python | wc -l`
	done
	python3.6 studies_stats.py --phs=$phs > results/log_$phs.txt &
done < $INPUT
IFS=$OLDIFS

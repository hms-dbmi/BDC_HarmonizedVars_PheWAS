INPUT=batch_list.txt
#INPUT=phs_list.txt
#CORES=`cat /proc/cpuinfo | grep processor | wc -l`
CORES=3
JOBS=`ps -df | grep studies_stats.py | wc -l`
echo "jobs at the beginning $JOBS"
echo "cores at the beginning: $CORES"

while IFS= read -r indic; do
        JOBS=`ps -df | grep studies_stats | wc -l`
        while [ "$[$CORES + 1]" -le "$JOBS" ]; do
                sleep 2
                JOBS=`ps -df | grep studies_stats.py | wc -l`
        done
        echo "launching process: $indic"
#        nohup python studies_stats.py --phs=$indic > studies_stats/by_phs/log_$indic.txt &
        nohup python studies_stats.py --batch_group=$indic > studies_stats/batch_group/logs/log_$indic.txt &
done < $INPUT


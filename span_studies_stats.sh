INPUT=env_variables/phs_list.txt
#INPUT=phs_list.txt
#CORES=`cat /proc/cpuinfo | grep processor | wc -l`
CORES=1
JOBS=`ps -df | grep studies_stats.py | wc -l`
echo "jobs at the beginning {$JOBS -1}"
echo "cores at the beginning: $CORES"

while IFS= read -r indic; do
        JOBS=`ps -df | grep studies_stats | wc -l`
        while [ "$[$CORES + 1]" -le "$JOBS" ]; do
                sleep 2
                JOBS=`ps -df | grep studies_stats.py | wc -l`
        done
        echo "launching process: $indic"
        echo $indic
       nohup python studies_stats.py --phs=$indic > logs/studies_stats/log_$indic.txt
done < $INPUT


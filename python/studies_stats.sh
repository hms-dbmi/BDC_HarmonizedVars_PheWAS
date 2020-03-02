#INPUT=../studies_info.csv
INPUT=phs_list.txt
#CORES=`cat /proc/cpuinfo | grep processor | wc -l`
CORES=8
JOBS=`ps -df | grep studies_stats | wc -l`
echo "jobs at the beginning $JOBS"
echo "cores at the beginning: $CORES"
#OLDIFS=$IFS
#IFS=','
#[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
#while read phs BDC_study_name official_study_name short_name harmonized phs_list ID_varName
while IFS= read -r phs; do
        JOBS=`ps -df | grep studies_stats | wc -l`
        while [ "$[$CORES + 1]" -le "$JOBS" ]; do
                echo "cores loop $CORES"
                echo "jobs loop $JOBS"
                sleep 2
                JOBS=`ps -df | grep studies_stats | wc -l`
        done
        echo "launching process: $phs"
        nohup python3.6 studies_stats.py --phs=$phs > results/log_$phs.txt &
done < $INPUT
#IFS=$OLDIFS


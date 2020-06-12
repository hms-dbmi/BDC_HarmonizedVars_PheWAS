#INPUT=batch_list.txt
INPUT=env_variables/phs_list.txt
#CORES=`cat /proc/cpuinfo | grep processor | wc -l`
CORES=1
JOBS=`ps -df | grep run_PheWAS.py | wc -l`
echo "jobs at the beginning $JOBS"
echo "cores at the beginning: $CORES"
#OLDIFS=$IFS
#IFS=','
#[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
#while read phs BDC_study_name official_study_name short_name harmonized phs_list ID_varName

#for f in $(ls ./studies_stats | grep "phs.*csv");
#!/bin/bash
while IFS= read -r phs;do
     #phs=${f::9}
        JOBS=`ps -df | grep run_PheWAS | wc -l`
        while [ "$[$CORES + 1]" -le "$JOBS" ]; do
                #echo "cores loop $CORES"
                #echo "jobs loop $JOBS"
                sleep 2
                JOBS=`ps -df | grep run_PheWAS.py | wc -l`
        done
       # echo $phs
        echo "launching process: $phs"
        nohup python run_PheWAS.py --phs=$phs > logs/PheWAS/log_PheWAS_$phs.txt
done < $INPUT
#IFS=$OLDIFS


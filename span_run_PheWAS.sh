#INPUT=batch_list.txt
INPUT=env_variables/list_phs_batchgroup.csv
#CORES=`cat /proc/cpuinfo | grep processor | wc -l`
CORES=44
JOBS=$(ps -df | grep run_PheWAS.py | wc -l)
echo "jobs at the beginning $JOBS"
echo "cores at the beginning: $CORES"
date_start_process=$( date +"%y%m%d_%H%M%S" )
path_results=results/"$date_start_process"
if [ ! -d "$path_results" ]
then
  mkdir -p "$path_results"
fi
#n=0
{
  read
  while IFS=, read -r phs batch_group
  do
     #phs=${f::9}
        JOBS=$(ps -df | grep run_PheWAS | wc -l)
        while [ "$[$CORES + 1]" -le "$JOBS" ]; do
#                echo "cores loop $CORES"
#                echo "jobs loop $JOBS"
                sleep 2
                JOBS=`ps -df | grep run_PheWAS.py | wc -l`
        done
#        if [ $n -le 5 ]
#         then
#          echo $n
#          echo $batch_group
#          echo $phs
#          echo $date_start_process
          echo "launching process: $phs $batch_group"
          nohup python run_PheWAS.py --phs="$phs" --batch_group=$batch_group --time_launched="$date_start_process" >> "$path_results"/logs_bash.txt &
          ((n=n+1))
#        fi
  done
} < $INPUT
echo "Done!"

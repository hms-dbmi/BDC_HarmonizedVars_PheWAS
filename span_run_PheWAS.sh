#INPUT=batch_list.txt
INPUT=env_variables/list_phs_batchgroup.csv
#CORES=`cat /proc/cpuinfo | grep processor | wc -l`
CORES=12
JOBS=$(ps -df | grep sleping_program.py | wc -l)
echo "jobs at the beginning $JOBS"
echo "cores at the beginning: $CORES"
date_start_process=$( date +"%y%m%d_%H%M%S" )
# date_start_process=010102_000000
path_results=results/"$date_start_process"
if [ ! -d "$path_results" ]
then
  mkdir -p "$path_results"
fi
n=0
{
  read
  while IFS=, read -r phs batch_group
  do
     #phs=${f::9}
        JOBS=$(( `ps -df | grep run_PheWAS.py | wc -l`-1 ))
        while [ "$CORES" -le "$JOBS" ]; do
                echo "jobs loop $JOBS"
                sleep 100
                JOBS=$(( `ps -df | grep run_PheWAS.py | wc -l`-1 ))
        done
          echo $n
#          echo $batch_group
#          echo $phs
#          echo $date_start_process
          # echo "launching process: $phs $batch_group"
#          nohup python sleeping_program.py &
          if [ ! -d "$path_results"/logs_bash/"$phs" ]
            then
              mkdir -p "$path_results"/logs_bash/"$phs"
          fi
          python run_PheWAS.py --phs="$phs" --batch_group=$batch_group --time_launched="$date_start_process" > "$path_results"/logs_bash/"$phs"/"$batch_group".txt &
          # python run_PheWAS.py --phs=phs000007 --batch_group=1 --time_launched=010101_00000 >> /run/user/1000/results_phewas/logs_bash.txt &
          ((n=n+1))
#        fi
  done
} < $INPUT
echo "Done!"

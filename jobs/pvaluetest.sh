#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-cannoj9
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2G

source ../../CTRNN/.venv/bin/activate
echo 'starting point'
python3 ../deep/deep_pipeline.py -a ../deep/results/factor1/ -d results/p_value_res_factor1.csv -n 100
echo 'Done'
sleep 5
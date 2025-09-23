#!/bin/sh

#sbatch_run_script.sh

#SBATCH -t 2:00:00
#SBATCH --partition=hacc
#SBATCH -A hpc-prf-acgasm
#SBATCH --exclusive 

source ~/repos/ops-hls-pact25-artifact/scripts/source_noctuna2_vitis_2022_2_ops.sh
module swap xilinx/u280 xilinx/vck5000

make run_hls_app

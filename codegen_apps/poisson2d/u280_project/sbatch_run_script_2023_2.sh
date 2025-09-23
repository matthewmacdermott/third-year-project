#!/bin/sh

#sbatch_run_script.sh

#SBATCH -t 2:00:00
#SBATCH --partition=fpga
#SBATCH --constraint=xilinx_u280_xrt2.16
#SBATCH -A hpc-prf-acgasm
#SBATCH --exclusive 

source ~/repos/ops-hls-pact25-artifact/scripts/source_noctuna2_vitis_2023_2_ops.sh

make run_hls_app

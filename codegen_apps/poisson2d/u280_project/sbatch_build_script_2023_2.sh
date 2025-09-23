#!/bin/sh

# sbatch_build_script.sh

#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -q fpgasynthesis
#SBATCH -A hpc-prf-acgasm
#SBATCH -p normal
#SBATCH --mail-type ALL
#SBATCH --mail-user beniel.thileepan@warwick.ac.uk

source ~/repos/ops-hls-pact25-artifact/scripts/source_noctuna2_vitis_2023_2_ops.sh

make clean
make

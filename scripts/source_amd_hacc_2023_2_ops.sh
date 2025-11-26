#!/bin/bash

#####################
# OPS setup script for Vitis 2023.2 for AMD HACC cluster at ETH Zurich.
# Author: Beniel.Thileepan@warwick.ac.uk
####################

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 1: ############# VITIS SPECIFIC SETUP ################
# Tool selection menu
# 1: Vitis 2023.2
# 2: Vitis 2024.1
# 3: Vitis 2024.2
# Change TOOL_SELECTION to select a different version
TOOL_SELECTION=1
source /opt/hdev/cli/enable/xrt << EOF
${TOOL_SELECTION}
EOF
source /opt/hdev/cli/enable/vivado << EOF
${TOOL_SELECTION}
EOF
source /opt/hdev/cli/enable/vitis << EOF
${TOOL_SELECTION}
EOF
# 2: ############# CPATH and INCLUDE SETUP #############

export CPATH=/usr/include/x86_64-linux-gnu/
VIVADO_INCLUDE_PATH=$(dirname $(dirname $(which vivado)))/include/
VITIS_INCLUDE_PATH=$(dirname $(dirname $(which vitis)))/include/
VITIS_HLS_INCLUDE_PATH=$(dirname $(dirname $(which vitis_hls)))/include/
export C_INCLUDE_PATH=${VITIS_INCLUDE_PATH}:${VIVADO_INCLUDE_PATH}:${VITIS_HLS_INCLUDE_PATH}
export CPP_INCLUDE_PATH=${VITIS_INCLUDE_PATH}:${VIVADO_INCLUDE_PATH}:${VITIS_HLS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${VITIS_INCLUDE_PATH}:${VIVADO_INCLUDE_PATH}:${VITIS_HLS_INCLUDE_PATH}

# 3: ############### OPS SPECIFICS SETUP ###############
# export OPS_GPP=/tools/Xilinx/Vitis/2023.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
# alias g++=/tools/Xilinx/Vitis/2023.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
export OPS_COMPILER=gnu

if [[ -n "$OPS_HLS_ARTIFACT_DIR" ]]; then
    export OPS_INSTALL_PATH=$OPS_HLS_ARTIFACT_DIR/OPS/ops
else
    export OPS_INSTALL_PATH=$SCRIPT_DIR/../OPS/ops
fi
export C_INCLUDE_PATH=${OPS_INSTALL_PATH}c/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${OPS_INSTALL_PATH}c/include/:$CPLUS_INCLUDE_PATH
export CPP_INCLUDE_PATH=${OPS_INSTALL_PATH}c/include/:$CPP_INCLUDE_PATH

# 4: ############ PYTHON VIRTUAL ENV SETUP #############

if [ -f ${OPS_INSTALL_PATH}/../ops_translator/ops_venv/bin/activate ]; then
    source ${OPS_INSTALL_PATH}/../ops_translator/ops_venv/bin/activate
else
    source ${OPS_INSTALL_PATH}/../ops_translator/setup_venv.sh
fi

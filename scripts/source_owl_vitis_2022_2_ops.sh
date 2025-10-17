#!/bin/bash

#####################
# OPS setup script for Vitis 2022.2 for KAUST HPC nodes.
# Author: Beniel.Thileepan@warwick.ac.uk
####################
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 1: ############# VITIS SPECIFIC SETUP ################

source /opt/xilinx/xrt/setup.sh

if [ -n "${PATH}" ]; then
  export PATH=/package/Xilinx/DocNav:$PATH
else
  export PATH=/package/Xilinx/DocNav
fi

export XILINX_VIVADO=/package/Xilinx/Vivado/2022.2
if [ -n "${PATH}" ]; then
  export PATH=/package/Xilinx/Vivado/2022.2/bin:$PATH
else
  export PATH=/package/Xilinx/Vivado/2022.2/bin
fi

export XILINX_VITIS=/package/Xilinx/Vitis/2022.2
if [ -n "${PATH}" ]; then
  export PATH=/package/Xilinx/Vitis/2022.2/bin:/package/Xilinx/Vitis/2022.2/gnu/microblaze/lin/bin:/package/Xilinx/Vitis/2022.2/gnu/arm/lin/bin:/package/Xilinx/Vitis/2022.2/gnu/microblaze/linux_toolchain/lin64_le/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-none/bin:/package/Xilinx/Vitis/2022.2/gnu/armr5/lin/gcc-arm-none-eabi/bin:/package/Xilinx/Vitis/2022.2/tps/lnx64/cmake-3.3.2/bin:/package/Xilinx/Vitis/2022.2/aietools/bin:$PATH
else
  export PATH=/package/Xilinx/Vitis/2022.2/bin:/package/Xilinx/Vitis/2022.2/gnu/microblaze/lin/bin:/package/Xilinx/Vitis/2022.2/gnu/arm/lin/bin:/package/Xilinx/Vitis/2022.2/gnu/microblaze/linux_toolchain/lin64_le/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin:/package/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-none/bin:/package/Xilinx/Vitis/2022.2/gnu/armr5/lin/gcc-arm-none-eabi/bin:/package/Xilinx/Vitis/2022.2/tps/lnx64/cmake-3.3.2/bin:/package/Xilinx/Vitis/2022.2/aietools/bin
fi

if [ -n "${PATH}" ]; then
  export PATH=/package/Xilinx/Model_Composer/2022.2/bin:$PATH
else
  export PATH=/package/Xilinx/Model_Composer/2022.2/bin
fi

export XILINX_HLS=/package/Xilinx/Vitis_HLS/2022.2
if [ -n "${PATH}" ]; then
  export PATH=/package/Xilinx/Vitis_HLS/2022.2/bin:$PATH
else
  export PATH=/package/Xilinx/Vitis_HLS/2022.2/bin
fi


# 2: ############# CPATH and INCLUDE SETUP #############

# export CPATH=
# /usr/include/x86_64-linux-gnu/
# VIVADO_INCLUDE_PATH=/tools/Xilinx/Vivado/2022.2/include/
# VITIS_INCLUDE_PATH=/tools/Xilinx/Vitis/2022.2/include/
# VITIS_HLS_INCLUDE_PATH=/tools/Xilinx/Vitis_HLS/2022.2/include/
# export C_INCLUDE_PATH=${VITIS_INCLUDE_PATH}:${VIVADO_INCLUDE_PATH}:${VITIS_HLS_INCLUDE_PATH}
# export CPP_INCLUDE_PATH=${VITIS_INCLUDE_PATH}:${VIVADO_INCLUDE_PATH}:${VITIS_HLS_INCLUDE_PATH}
# export CPLUS_INCLUDE_PATH=${VITIS_INCLUDE_PATH}:${VIVADO_INCLUDE_PATH}:${VITIS_HLS_INCLUDE_PATH}

# 3: ############### OPS SPECIFICS SETUP ###############
module load GCC/12.2.0
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

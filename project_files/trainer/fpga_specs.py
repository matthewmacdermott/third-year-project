"""FPGA resource constraints for the U280 target.

This module provides U280 total-device and per-SLR resource capacities and
helpers to normalize utilization.
"""

from typing import Dict, NamedTuple


class FPGAResources(NamedTuple):
    """Resource capacity specification for an FPGA device."""
    bram: int      # BRAM_18K blocks
    dsp: int       # DSP slices
    ff: int        # Flip-flops
    lut: int       # Look-up tables


class FPGASpec(NamedTuple):
    """Complete FPGA specification with device and SLR limits."""
    total: FPGAResources    # Total device capacity
    slr: FPGAResources      # Single SLR capacity
    num_slrs: int           # Number of SLRs on device


# AMD/Xilinx Alveo U280
U280_SPEC = FPGASpec(
    total=FPGAResources(
        bram=4032,      # Total BRAM_18K
        dsp=9024,       # Total DSP slices
        ff=2607360,     # Total flip-flops
        lut=1303680,    # Total LUTs
    ),
    slr=FPGAResources(
        bram=1344,      # BRAM_18K per SLR
        dsp=2664,       # DSP available in the dynamic pblock per SLR on U280
        ff=869120,      # FF per SLR
        lut=434560,     # LUT per SLR
    ),
    num_slrs=3
)

U280_MAXI_WIDTH_BITS = 512


def calculate_resource_utilization(resources: Dict[str, int], 
                                   fpga_spec: FPGASpec = U280_SPEC) -> Dict[str, float]:
    """Calculate normalized resource utilization (0-1 scale).
    
    Args:
        resources: Dictionary with resource usage (bram, dsp, ff, lut)
        fpga_spec: Target FPGA specification
        
    Returns:
        Dictionary with normalized utilization per resource type
        
    Raises:
        ValueError: If FPGA spec has zero capacity for any resource
    """
    util = {}
    for key in ['bram', 'dsp', 'ff', 'lut']:
        used = resources.get(key, 0)
        total = getattr(fpga_spec.total, key)
        if total <= 0:
            raise ValueError(
                f"Invalid FPGA spec: {key} capacity is {total}. "
                f"All resource capacities must be positive."
            )
        util[key] = used / total
    return util


def calculate_slr_utilization(resources: Dict[str, int],
                              fpga_spec: FPGASpec = U280_SPEC) -> Dict[str, float]:
    """Calculate per-SLR resource utilization.
    
    Args:
    resources: Dictionary with resource usage (bram, dsp, ff, lut)
        fpga_spec: Target FPGA specification

    Returns:
        Dictionary with SLR utilization per resource type

    Raises:
        ValueError: If FPGA spec has zero SLR capacity for any resource
    """
    util = {}
    for key in ['bram', 'dsp', 'ff', 'lut']:
        used = resources.get(key, 0)
        slr_capacity = getattr(fpga_spec.slr, key)
        if slr_capacity <= 0:
            raise ValueError(
                f"Invalid FPGA spec: SLR {key} capacity is {slr_capacity}. "
                f"All SLR resource capacities must be positive."
            )
        util[key] = used / slr_capacity
    return util


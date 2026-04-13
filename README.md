# HLS Stencil Optimiser: Setup and Run Guide

## 1. Prerequisites

- Linux shell
- Access to the PC2 Noctua 2 system
- Git access to `git@github.com:matthewmacdermott/third-year-project.git`
- Xilinx/Vitis environment available on your system
- Python 3.8+

## 2. One-time setup

Run the following commands on the Noctua2 system:

```bash
project_name="3yp"
git clone git@github.com:matthewmacdermott/third-year-project.git $project_name
cd $project_name
export OPS_HLS_ARTIFACT_DIR=$(pwd)
git submodule update --recursive --init
source scripts/source_noctuna2_vitis_2023_2_ops.sh
cd $OPS_INSTALL_PATH/hls
make
```

What these commands do:

1. Clones the repository into `3yp`
2. Sets `OPS_HLS_ARTIFACT_DIR` to the repo root
3. Initializes git submodules
4. Sources `scripts/source_noctuna2_vitis_2023_2_ops.sh`
5. Builds OPS HLS components via `make` in `$OPS_INSTALL_PATH/hls`

After setup completes, return to the project root (`3yp`) before launching
the UI:

```bash
cd "$OPS_HLS_ARTIFACT_DIR"
```

## 3. Launch the UI

From the project root:

```bash
./run_system.sh
```

This launcher will:

- Source the Xilinx/OPS environment if needed
- Pick a Python >= 3.8
- Activate a local virtualenv if one exists
- Install missing Python dependencies from `project_files/requirements.txt` when needed
- Start the interactive UI (`project_files.system.main`)

## 4. First run flow in UI

Once the UI opens, the common path is:

1. `Train a new model` (or pick an existing one)
2. `Run recommendation -> build -> results`
3. Choose model and stencil
4. Choose strategy (`balanced`, `resource`, `performance`)
5. Review top recommendations
6. Optionally apply config and submit build/probe jobs

## 5. If `run_system.sh` is not executable

```bash
chmod +x run_system.sh
./run_system.sh
```

## 6. Troubleshooting

- Environment not loaded:
  - Run `source scripts/source_noctuna2_vitis_2023_2_ops.sh`
- Missing Python packages:
  - `./run_system.sh` auto-installs from `project_files/requirements.txt`

## 7. Notes

- The setup script is intended for fresh setup on a new workspace.
- If you already have this repository checked out and submodules initialized, you can usually skip setup and run `./run_system.sh` directly.

## 8. Running Generated HLS xclbins

All detailed instructions for running stencil applications after building with the UI are in `codegen_apps/README.md`

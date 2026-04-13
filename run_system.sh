#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_system.sh  –  Entry point for the HLS Stencil Optimiser
#
# Usage:
#   ./run_system.sh [--root /path/to/project] [extra python args...]
#
# Run from any directory; the script resolves paths automatically.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# run_system.sh lives at the project root
PROJECT_ROOT="$SCRIPT_DIR"
PROJECT_FILES_DIR="$PROJECT_ROOT/project_files"

# ── Source Vitis / OPS environment if not already loaded ─────────────────────
if [ -z "${XILINX_VITIS:-}" ]; then
    OPS_SETUP="$PROJECT_ROOT/scripts/source_noctuna2_vitis_2023_2_ops.sh"
    if [ -f "$OPS_SETUP" ]; then
        echo "Sourcing Xilinx / OPS environment: $OPS_SETUP"
        # Temporarily disable nounset (-u) — the setup script references
        # variables like C_INCLUDE_PATH that may not yet be defined.
        set +u
        # shellcheck source=/dev/null
        source "$OPS_SETUP"
        set -u
    else
        echo "Warning: XILINX_VITIS not set and setup script not found at $OPS_SETUP" >&2
        echo "         Run: source scripts/source_noctuna2_vitis_2023_2_ops.sh" >&2
    fi
fi

# ── Python interpreter (prefer >= 3.8 for rustworkx / torch compatibility) ───
PYTHON=""
for _candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
    if command -v "$_candidate" &>/dev/null; then
        _ver=$("$_candidate" -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])' 2>/dev/null)
        if [ -n "$_ver" ] && [ "$_ver" -ge 308 ]; then
            PYTHON="$_candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: no Python >= 3.8 found on PATH." >&2
    echo "       Load the appropriate module or activate a conda/venv environment first." >&2
    exit 1
fi

PYTHON_VER=$("$PYTHON" -c 'import sys; print(sys.version_info[:2])')
echo "Using $PYTHON  (version $PYTHON_VER)"

# ── Optional: activate a venv if one exists ──────────────────────────────────
for venv_path in \
    "$PROJECT_FILES_DIR/.venv" \
    "$PROJECT_ROOT/.venv" \
    "$HOME/.venv/ops_hls"
do
    if [ -f "$venv_path/bin/activate" ]; then
        echo "Activating venv: $venv_path"
        # shellcheck source=/dev/null
        source "$venv_path/bin/activate"
        break
    fi
done

# ── Check / auto-install ML dependencies ────────────────────────────────────
REQUIREMENTS="$PROJECT_FILES_DIR/requirements.txt"

MISSING=""
for pkg in numpy torch; do
    if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo ""
    echo "  Missing packages:$MISSING"
    echo ""

    # ── Bootstrap pip if it is absent ────────────────────────────────────────
    if ! "$PYTHON" -m pip --version &>/dev/null; then
        echo "  pip not found for $PYTHON – bootstrapping with ensurepip ..."
        if "$PYTHON" -m ensurepip --upgrade 2>/dev/null; then
            echo "  pip bootstrapped."
        else
            echo "  ensurepip unavailable.  Trying get-pip.py ..."
            _GETPIP=$(mktemp /tmp/get-pip-XXXXXX.py)
            if curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$_GETPIP" 2>/dev/null \
               || wget -qO "$_GETPIP" https://bootstrap.pypa.io/get-pip.py 2>/dev/null; then
                "$PYTHON" "$_GETPIP" --user
            fi
            rm -f "$_GETPIP"
        fi
    fi

    # Final check – bail with a clear message if pip is still missing
    if ! "$PYTHON" -m pip --version &>/dev/null; then
        echo ""
        echo "  ERROR: could not find or install pip for $PYTHON."
        echo "  Please run one of:"
        echo "    $PYTHON -m ensurepip --upgrade"
        echo "    curl -fsSL https://bootstrap.pypa.io/get-pip.py | $PYTHON"
        echo ""
        exit 1
    fi

    # Decide pip flag: no --user inside a venv
    IN_VENV=$("$PYTHON" -c 'import sys; print(int(sys.prefix != sys.base_prefix))')
    if [ "$IN_VENV" -eq 1 ]; then
        PIP_FLAGS=""
    else
        PIP_FLAGS="--user"
    fi

    echo "  Installing from $REQUIREMENTS ..."
    "$PYTHON" -m pip install $PIP_FLAGS -r "$REQUIREMENTS"
    if [ $? -ne 0 ]; then
        echo ""
        echo "  pip install failed. Please install manually:"
        echo "    $PYTHON -m pip install $PIP_FLAGS -r $REQUIREMENTS"
        echo ""
        # Continue anyway – non-recommender features still work
    else
        echo "  Dependencies installed."
    fi
    echo ""
fi

# ── Launch ───────────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m project_files.system.main \
    --root "$PROJECT_ROOT" \
    "$@"

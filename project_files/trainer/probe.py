#!/usr/bin/env python3
"""
probe.py — Run a single stencil + config through the synthesis pipeline and
inspect the reward signal.

"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

# ── make sure we can import from the trainer package ──────────────────────────
TRAINER_DIR = Path(__file__).resolve().parent
PROJ_DIR    = TRAINER_DIR.parent
REPO_DIR    = PROJ_DIR.parent
sys.path.insert(0, str(TRAINER_DIR))
sys.path.insert(0, str(PROJ_DIR))

# ── Optionally change cwd so relative paths inside the env resolve ─────────────
os.chdir(REPO_DIR)

from hls_environment import ParallelHLSEnvironment

# ── Default / canonical param values (used when a param is not overridden) ────
_PARAM_DEFAULTS = {
    "vector_factor":               4,
    "mem_vector_factor":           4,
    "iter_par_factor":             1,
    "maxi_depth":               4096,
    "maxi_read_burst_length":    128,
    "maxi_write_burst_length":   128,
    "num_read_outstanding":        2,
    "num_write_outstanding":       2,
    "axis_interconnect_buff_size": 1024,
    "hls_interconnect_buff_size":  64,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe a single stencil + config through the HLS pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    p.add_argument("app", nargs="?", help="App / stencil name, e.g. laplace2d")

    # Actions
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list",        action="store_true", help="List available apps and exit")
    g.add_argument("--show-config", action="store_true", help="Print the on-disk config for the app and exit")
    g.add_argument("--dry-run",     action="store_true", help="Validate + print state; skip synthesis")

    # Config source — JSON blob or per-param shorthands
    p.add_argument("--config", type=str, default=None,
                   help="Full config as a JSON object string")

    # Per-param shorthands
    p.add_argument("--vf",    type=int, dest="vector_factor",               metavar="N")
    p.add_argument("--mvf",   type=int, dest="mem_vector_factor",           metavar="N")
    p.add_argument("--ipf",   type=int, dest="iter_par_factor",             metavar="N")
    p.add_argument("--depth", type=int, dest="maxi_depth",                  metavar="N")
    p.add_argument("--rb",    type=int, dest="maxi_read_burst_length",      metavar="N")
    p.add_argument("--wb",    type=int, dest="maxi_write_burst_length",     metavar="N")
    p.add_argument("--ro",    type=int, dest="num_read_outstanding",        metavar="N")
    p.add_argument("--wo",    type=int, dest="num_write_outstanding",       metavar="N")
    p.add_argument("--axis",  type=int, dest="axis_interconnect_buff_size", metavar="N")
    p.add_argument("--hls",   type=int, dest="hls_interconnect_buff_size",  metavar="N")

    # Misc
    p.add_argument("--timeout", type=int, default=3600,
                   help="Total synthesis timeout in seconds (default: 3600)")
    p.add_argument("--no-stencil-features", action="store_true",
                   help="Disable stencil feature loading (faster startup)")

    return p.parse_args()


def build_config(args) -> dict:
    """Merge defaults → JSON blob → CLI shorthands (highest priority wins)."""
    config = dict(_PARAM_DEFAULTS)

    if args.config:
        try:
            override = json.loads(args.config)
        except json.JSONDecodeError as e:
            sys.exit(f"[probe] --config JSON parse error: {e}")
        config.update(override)

    # Per-param shorthands
    shorthand_keys = [
        "vector_factor", "mem_vector_factor", "iter_par_factor",
        "maxi_depth", "maxi_read_burst_length", "maxi_write_burst_length",
        "num_read_outstanding", "num_write_outstanding",
        "axis_interconnect_buff_size", "hls_interconnect_buff_size",
    ]
    for k in shorthand_keys:
        v = getattr(args, k, None)
        if v is not None:
            config[k] = v

    return config


def print_separator(char="─", width=72):
    print(char * width)


def print_metrics(metrics: dict):
    print_separator()
    print("  RAW SYNTHESIS METRICS")
    print_separator()
    skip = {"error_message"}
    for k, v in sorted(metrics.items()):
        if k in skip:
            continue
        if isinstance(v, float):
            print(f"  {k:<40s} {v:.6g}")
        else:
            print(f"  {k:<40s} {v!r}")
    if "error_message" in metrics and metrics["error_message"]:
        print(f"\n  error_message: {metrics['error_message']}")
    print_separator()


def print_state(state, label="STATE VECTOR"):
    print_separator()
    print(f"  {label}  (dim={len(state)})")
    print_separator()
    config_block  = state[:10]
    stencil_block = state[10:]
    print(f"  config  [0:10]  : {' '.join(f'{v:7.4f}' for v in config_block)}")
    n_pts = (len(stencil_block) - 3) // 3
    print(f"  stencil[10:{10+len(stencil_block)}] : global={' '.join(f'{v:.3f}' for v in stencil_block[:3])}")

    # Print the full padded stencil payload so logs can be copied verbatim.
    print("  stencil triplets (dx,dy,dz):")
    triplets = stencil_block[3:]
    for i in range(n_pts):
        base = i * 3
        dx, dy, dz = triplets[base:base + 3]
        print(f"    [{i:02d}] ({dx:+.4f}, {dy:+.4f}, {dz:+.4f})")
    print_separator()


def main():
    args = parse_args()

    enable_stencil = not args.no_stencil_features
    print("[probe] Initialising environment …")
    env = ParallelHLSEnvironment(
        apps_base_dir=str(REPO_DIR / "codegen_apps"),
        max_parallel_jobs=1,
        enable_stencil_features=enable_stencil,
    )

    # ── --list ─────────────────────────────────────────────────────────────────
    if args.list:
        print("\nAvailable apps:")
        for app in sorted(env.available_apps, key=lambda a: a["app_name"]):
            print(f"  {app['app_name']:<30s}  {app['project_dir']}")
        return

    # ── Require app name from here on ─────────────────────────────────────────
    if not args.app:
        sys.exit("[probe] Provide an app name or use --list.  Try: python probe.py --list")

    app_name = args.app
    matching = [a for a in env.available_apps if a["app_name"] == app_name]
    if not matching:
        available = sorted(set(a["app_name"] for a in env.available_apps))
        sys.exit(
            f"[probe] Unknown app '{app_name}'.\n"
            f"  Available: {', '.join(available)}"
        )
    app_info = matching[0]

    # ── --show-config ──────────────────────────────────────────────────────────
    if args.show_config:
        with open(app_info["config_file"]) as f:
            disk_config = json.load(f)
        print(f"\n  On-disk config for {app_name}  ({app_info['config_file']}):")
        print(json.dumps(disk_config, indent=4))
        return

    # ── Build config ──────────────────────────────────────────────────────────
    config = build_config(args)

    print()
    print_separator("═")
    print(f"  PROBE:  {app_name}")
    print_separator("═")
    print("  Config:")
    for k, v in sorted(config.items()):
        default_marker = " (default)" if v == _PARAM_DEFAULTS.get(k) else ""
        print(f"    {k:<40s} = {v}{default_marker}")

    # ── Pre-validate ──────────────────────────────────────────────────────────
    print()
    valid, reason = env._is_config_valid(config, env.u280_maxi_width_bits, env.param_ranges)
    if not valid:
        print(f"  Pre-validation FAILED: {reason}")
        print("  Synthesis skipped.")
        return
    print(f"  Pre-validation passed")

    # ── State vector ──────────────────────────────────────────────────────────
    state = env._create_state(config, app_info.get("stencil_features"))
    print_state(state)

    if args.dry_run:
        print("[probe] --dry-run: stopping before synthesis.\n")
        return

    # ── Synthesis ─────────────────────────────────────────────────────────────
    print(f"\n[probe] Running synthesis (timeout={args.timeout}s) …\n")
    metrics = env._run_synthesis(app_info, config)

    print()
    print_metrics(metrics)

    # ── Reward ────────────────────────────────────────────────────────────────
    if metrics.get("synthesis_success"):
        print("\n[probe] Calculating reward …")
        reward = env._calculate_reward(metrics, config, app_info)
        print()
        print_separator("═")
        print("  REWARD SUMMARY")
        print_separator("═")
        labels = ["runtime", "bram", "dsp", "lut", "ff", "freq"]
        for label, value in zip(labels, reward):
            bar_width = 30
            filled = int(max(0.0, value) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  {label:<10s}  [{bar}]  {value:+.4f}")
        print_separator()
        print(f"  total (mean)            = {reward.mean():.4f}")
        print(f"  total (sum)             = {reward.sum():.4f}")
        print_separator("═")
    else:
        err = metrics.get("error_message", "unknown error")
        print(f"\n[probe] Synthesis FAILED: {err}")

    print()


if __name__ == "__main__":
    main()

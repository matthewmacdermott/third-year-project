#!/usr/bin/env python3
"""Entry point for the trainer SLURM node in multi-node DQN training.

This script is launched by the trainer sbatch job.  It drives
ParallelPyTorchRLOptimizer.train_from_store() which continuously reads
experiences written by worker nodes and performs gradient updates.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Multi-node DQN trainer node")
    parser.add_argument("--store",          required=True,  help="Path to experience_store.db")
    parser.add_argument("--checkpoint",     required=True,  help="Path for shared latest.pth")
    parser.add_argument("--run-dir",        required=True,  help="Directory for models/results")
    parser.add_argument("--max-time",       type=float, required=True, help="Wall-clock hours")
    parser.add_argument("--save-freq",      type=int,   default=20)
    parser.add_argument("--poll-interval",  type=float, default=30.0, help="Seconds between store polls")
    parser.add_argument("--load-model",       type=str,   default=None, help="Pre-trained model path")
    parser.add_argument("--reset-epsilon",    type=float, default=None, help="Override epsilon after loading model (e.g. 0.3)")
    parser.add_argument("--pretrain-store",   type=str,   nargs="+",   default=None, help="Path(s) to existing experience_store.db for offline pre-training")
    parser.add_argument("--pretrain-epochs",  type=int,   default=3,   help="Number of epochs for offline pre-training (default: 3)")
    parser.add_argument("--early-stopping-patience",   type=int,   default=100)
    parser.add_argument("--early-stopping-min-delta",  type=float, default=0.01)
    args = parser.parse_args()

    # We need a minimal apps_dir so ParallelHLSEnvironment can initialise
    # (it reads param ranges and app list at construction time).
    # The trainer node never runs synthesis itself, so no valid FPGA env is needed.
    run_dir  = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Infer apps_dir relative to this file's location
    trainer_dir = Path(__file__).parent
    project_dir = trainer_dir.parent
    apps_dir    = project_dir.parent / "codegen_apps"

    print("=== Multi-node DQN Trainer Node ===")
    print(f"  store:      {args.store}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  run_dir:    {run_dir}")
    print(f"  max_time:   {args.max_time}h")
    print(f"  apps_dir:   {apps_dir}")

    from optimizer import ParallelPyTorchRLOptimizer

    optimizer = ParallelPyTorchRLOptimizer(
        apps_base_dir=str(apps_dir),
        max_parallel_jobs=1,   # trainer never synthesises
        batch_size=1,
        run_dir=str(run_dir),
    )

    if args.pretrain_store:
        print(f"  Pre-training on {len(args.pretrain_store)} store(s), {args.pretrain_epochs} epochs")
        optimizer.pretrain_from_stores(args.pretrain_store, n_epochs=args.pretrain_epochs)
        pretrained_path = str(run_dir / "pretrained.pth")
        optimizer.agent.save(pretrained_path)
        print(f"  Saved pre-trained model: {pretrained_path}")

    if args.load_model:
        optimizer.agent.load(args.load_model)
        print(f"  Loaded model: {args.load_model}")
        if args.reset_epsilon is not None:
            optimizer.agent.epsilon = args.reset_epsilon
            print(f"  Epsilon reset to: {args.reset_epsilon}")

    optimizer.train_from_store(
        store_path=args.store,
        shared_checkpoint=args.checkpoint,
        max_time_hours=args.max_time,
        save_freq=args.save_freq,
        poll_interval_s=args.poll_interval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


if __name__ == "__main__":
    main()

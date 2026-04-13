#!/usr/bin/env python3
"""Parallel PyTorch Deep RL HLS Synthesis Optimizer."""

import argparse
from optimizer import ParallelPyTorchRLOptimizer


def main():
    """Main function for parallel PyTorch RL optimization."""
    print("=== PyTorch Deep RL HLS Optimizer ===\n")
    
    parser = argparse.ArgumentParser(description='PyTorch Deep RL HLS Optimization')
    parser.add_argument('--num-runs', type=int, default=None, help='Total synthesis runs')
    parser.add_argument('--max-time', type=float, default=None, help='Max training time in hours')
    parser.add_argument('--save-freq', type=int, default=10, help='Model check frequency')
    parser.add_argument('--batch-size', type=int, default=16, help='Parallel synthesis jobs')
    parser.add_argument('--load-model', type=str, default=None, help='Pre-trained model path')
    parser.add_argument('--run-dir', type=str, default=None, help='Directory for models/results output')
    parser.add_argument('--early-stopping-patience', type=int, default=100, 
                       help='Early stopping patience')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.01,
                       help='Minimum improvement threshold')
    
    args = parser.parse_args()
    
    if args.num_runs is None and args.max_time is None:
        parser.error('Must specify either --num-runs or --max-time (or both)')
    
    from pathlib import Path
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(__file__).parent.parent / "trainer" / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    config = {
        'timestamp': timestamp,
        'num_runs': args.num_runs,
        'max_time': args.max_time,
        'save_freq': args.save_freq,
        'batch_size': args.batch_size,
        'load_model': args.load_model,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run directory: {run_dir}\n")
    
    apps_dir = Path(__file__).parent.parent.parent / "codegen_apps"
    
    optimizer = ParallelPyTorchRLOptimizer(
        apps_base_dir=str(apps_dir),
        max_parallel_jobs=args.batch_size,
        batch_size=args.batch_size,
        run_dir=str(run_dir)
    )
    
    if args.load_model:
        print(f"Loading model: {args.load_model}")
        optimizer.agent.load(args.load_model)
    
    if args.num_runs and args.max_time:
        print(f"\nTraining: {args.num_runs} runs or {args.max_time}h (whichever first)")
    elif args.num_runs:
        print(f"\nTraining: {args.num_runs} runs")
    else:
        print(f"\nTraining: {args.max_time}h")
    print(f"Batch size: {args.batch_size}\n")
    
    run_rewards = optimizer.train(
        num_runs=args.num_runs,
        save_freq=args.save_freq,
        max_time_hours=args.max_time,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta
    )
    
    optimizer.save_results()

    # Always save a final model so there is always something in models/
    final_path = optimizer.model_save_dir / f"dqn_final_{timestamp}.pth"
    optimizer.agent.save(str(final_path))
    print(f"Final model saved: {final_path}")

    print(f"\n=== Training Summary ===")
    print(f"Total runs: {len(run_rewards)}")
    print(f"Avg reward: {sum(run_rewards)/len(run_rewards):.3f}")
    print(f"Best reward: {max(run_rewards):.3f}")
    successful = sum(1 for h in optimizer.env.synthesis_history if h['metrics'].get('synthesis_success', False))
    print(f"Successful: {successful}")
    if optimizer.best_model_path:
        print(f"Best model: {optimizer.best_model_path.name} ({optimizer.best_avg_reward:.4f})")
    
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()

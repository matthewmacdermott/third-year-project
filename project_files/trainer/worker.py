#!/usr/bin/env python3
"""Worker node for multi-node DQN training.

Each SLURM worker job runs this script.  It:
  1. Loads the latest shared checkpoint (if any).
  2. Samples a batch of apps and chooses configs via epsilon-greedy.
  3. Runs Vitis HLS synthesis in parallel.
  4. Writes every (stencil_state, action, reward) tuple to the shared
     ExperienceStore on disk.
  5. Repeats until --num-runs is exhausted or --max-time is reached.

The worker never calls agent.replay() — gradient updates are the trainer's job.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Trainer-package imports — this file lives alongside them
from hls_environment import ParallelHLSEnvironment
from dqn_agent import DQNAgent
from experience_store import ExperienceStore


def _build_agent(env: ParallelHLSEnvironment, epsilon: float = 1.0) -> DQNAgent:
    return DQNAgent(
        stencil_dim=env.stencil_dim,
        config_dim=env.config_dim,
        action_dim=env.action_dim,
        decode_action_fn=env.get_config_vector,
        is_valid_fn=lambda idx: env._is_config_valid(
            env.get_config_from_action(idx)
        )[0],
        learning_rate=1e-3,
        gamma=0.95,
        epsilon=epsilon,
        epsilon_decay=0.99993,
        epsilon_min=0.01,
        memory_size=1,       # worker does not replay — minimal buffer
        batch_size=32,
        target_update_freq=999999,
    )


def run_worker(
    apps_dir: str,
    store_path: str,
    checkpoint_path: str,
    *,
    num_runs: int = None,
    max_time_hours: float = None,
    batch_size: int = 4,
    worker_id: int = 0,
):
    print(f"=== Multi-node Worker {worker_id} ===")
    print(f"  store:      {store_path}")
    print(f"  checkpoint: {checkpoint_path}")

    store = ExperienceStore(store_path)
    env   = ParallelHLSEnvironment(apps_dir, max_parallel_jobs=batch_size)
    agent = _build_agent(env)

    # Build a per-app deduplication cache from everything already in the store.
    # This prevents re-synthesising configs we already have results for.
    seen_actions: dict = store.seen_actions_by_app()
    total_seen = sum(len(v) for v in seen_actions.values())
    print(f"  Dedup cache loaded: {total_seen} seen (app, action) pairs across "
          f"{len(seen_actions)} app(s)")

    # Load the latest shared checkpoint if present
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        agent.load(str(ckpt))
        print(f"  Loaded checkpoint (ε={agent.epsilon:.3f}, step={agent.training_step})")
    else:
        print("  No checkpoint found — starting with random policy")

    start_time = time.time()
    max_time_s = max_time_hours * 3600 if max_time_hours else None
    total_runs = 0

    while True:
        # Time / run budget check
        if max_time_s and (time.time() - start_time) >= max_time_s:
            print(f"\nWorker {worker_id}: time budget reached after {total_runs} runs")
            break
        if num_runs is not None and total_runs >= num_runs:
            print(f"\nWorker {worker_id}: run budget reached ({total_runs})")
            break

        # Reload checkpoint periodically so we benefit from the trainer's updates
        if ckpt.exists() and total_runs > 0 and total_runs % (batch_size * 4) == 0:
            try:
                agent.load(str(ckpt))
                print(f"[Worker {worker_id}] Reloaded checkpoint (ε={agent.epsilon:.3f})")
            except Exception as e:
                print(f"[Worker {worker_id}] Checkpoint reload failed (ignored): {e}")

        current_batch = batch_size
        if num_runs is not None:
            current_batch = min(batch_size, num_runs - total_runs)

        print(f"\n[Worker {worker_id}] Batch starting (runs {total_runs}‒{total_runs + current_batch - 1})")

        # 1. Sample apps
        batch_apps = env.sample_batch_apps(current_batch)

        # 2. Build stencil context states and pick actions
        context_states = [
            env._create_state({}, app.get('stencil_features'))[env.config_dim:]
            for app in batch_apps
        ]
        batch_actions = agent.act_batch(context_states, training=True, verbose_first=True)

        # Deduplicate: re-sample any action already seen for that app (up to 10 tries).
        for i, (action, app) in enumerate(zip(batch_actions, batch_apps)):
            app_name = app['app_name']
            app_seen = seen_actions.get(app_name, set())
            if action in app_seen:
                state = context_states[i]
                for _ in range(10):
                    new_action = agent.act(state, training=True)
                    if new_action not in app_seen:
                        batch_actions[i] = new_action
                        break
                # If all retries were seen (extremely unlikely at low coverage),
                # keep the duplicate — training on it again is harmless.

        # 3. Run synthesis
        action_configs = [
            (action, env.get_config_from_action(action), batch_apps[i])
            for i, action in enumerate(batch_actions)
        ]
        batch_results = env.run_parallel_synthesis_batch(action_configs)

        # 4. Write each experience to the shared store.
        # Results arrive in completion order, so never zip with batch_apps.
        for result in batch_results:
            app_name = result.get('app_name', 'unknown')
            # Update local dedup cache immediately so within-batch duplicates
            # are also avoided across iterations.
            seen_actions.setdefault(app_name, set()).add(result['action'])
            stencil_state = result['stencil_state']
            action        = result['action']
            reward        = np.asarray(result['reward'], dtype=np.float32)
            cfg           = result['config']

            row_id = store.push(
                app_name=app_name,
                stencil_features=stencil_state,
                action=action,
                config=cfg,
                reward=reward,
                epsilon=agent.epsilon,
            )

            rw_str = f"[{', '.join(f'{v:.3f}' for v in reward)}]"
            print(
                f"[Worker {worker_id}] store row={row_id}"
                f"  {app_name}  V={cfg.get('vector_factor','?')}"
                f"  MVF={cfg.get('mem_vector_factor','?')}"
                f"  rewards={rw_str}"
            )

        total_runs += current_batch

    print(f"\nWorker {worker_id} done. Total synthesis runs: {total_runs}")
    print(f"Store now contains {store.count()} experiences total")


def main():
    parser = argparse.ArgumentParser(description="Multi-node synthesis worker")
    parser.add_argument("--apps-dir",       required=True)
    parser.add_argument("--store",          required=True,  help="Path to experience_store.db")
    parser.add_argument("--checkpoint",     required=True,  help="Path to shared latest.pth")
    parser.add_argument("--num-runs",       type=int,   default=None)
    parser.add_argument("--max-time",       type=float, default=None, help="Hours")
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--worker-id",      type=int,   default=0)
    args = parser.parse_args()

    if args.num_runs is None and args.max_time is None:
        parser.error("Specify --num-runs or --max-time")

    run_worker(
        apps_dir=args.apps_dir,
        store_path=args.store,
        checkpoint_path=args.checkpoint,
        num_runs=args.num_runs,
        max_time_hours=args.max_time,
        batch_size=args.batch_size,
        worker_id=args.worker_id,
    )


if __name__ == "__main__":
    main()

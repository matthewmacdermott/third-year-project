import json
import time
import numpy as np
from pathlib import Path
from typing import List, Optional

from hls_environment import ParallelHLSEnvironment
from dqn_agent import DQNAgent
from experience import Experience


class ParallelPyTorchRLOptimizer:
    """Main optimizer class using parallel PyTorch DQN."""
    
    def __init__(self, apps_base_dir: str = "codegen_apps",
                 model_save_dir: str = None,
                 max_parallel_jobs: int = None, batch_size: int = 4,
                 run_dir: str = None):
        
        self.env = ParallelHLSEnvironment(apps_base_dir,
                                         max_parallel_jobs=max_parallel_jobs)
        self.synthesis_batch_size = batch_size
        
        self.agent = DQNAgent(
            stencil_dim=self.env.stencil_dim,
            config_dim=self.env.config_dim,
            action_dim=self.env.action_dim,
            decode_action_fn=self.env.get_config_vector,
            is_valid_fn=lambda idx: self.env._is_config_valid(
                self.env.get_config_from_action(idx)
            )[0],
            learning_rate=1e-3,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.999990,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=32,
            target_update_freq=100
        )
        
        if run_dir:
            self.run_dir = Path(run_dir)
            self.model_save_dir = self.run_dir / "models"
            self.results_dir = self.run_dir / "results"
        else:
            self.run_dir = Path(".")
            self.model_save_dir = Path(model_save_dir) if model_save_dir else Path("models")
            self.results_dir = Path("parallel_pytorch_results")
        
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_rewards = []
        self.run_losses = []
        self.best_avg_reward = float('-inf')
        self.best_model_path = None
        self.patience_counter = 0
        
        print(f"Optimizer initialized: {self.env.action_dim} actions, {self.env.state_dim} state dims")
    
    def train(self, num_runs: int = None, save_freq: int = 10, max_time_hours: float = None,
              early_stopping_patience: int = 100, early_stopping_min_delta: float = 0.01) -> List[float]:
        """Train the DQN agent with early stopping."""
        import time
        
        start_time = time.time()
        max_time_seconds = max_time_hours * 3600 if max_time_hours else None
        
        print(f"Starting parallel DQN training...")
        if num_runs:
            print(f"Target runs: {num_runs}")
        if max_time_hours:
            print(f"Time budget: {max_time_hours} hours")
        print(f"Batch size: {self.synthesis_batch_size}")
        print(f"Early stopping: patience={early_stopping_patience}")
        
        reward_window_size = 20
        
        run = 0
        while True:
            if max_time_seconds:
                elapsed = time.time() - start_time
                if elapsed >= max_time_seconds:
                    print(f"\nTime budget reached ({elapsed/3600:.1f}h)")
                    print(f"Completed {run} runs")
                    break
            
            if num_runs and run >= num_runs:
                print(f"\nCompleted {num_runs} runs")
                break
            
            if num_runs:
                remaining_runs = num_runs - run
                current_batch_size = min(self.synthesis_batch_size, remaining_runs)
            else:
                current_batch_size = self.synthesis_batch_size
            
            print(f"\n=== Batch {run//self.synthesis_batch_size + 1} (Runs {run}-{run + current_batch_size - 1}) ===")

            # 1. Sample apps for this batch — one app per job
            batch_apps = self.env.sample_batch_apps(current_batch_size)

            # 2. Build stencil-only context states so the agent can choose the
            #    right config for each app's stencil.  The pair network accepts
            #    stencil features (84-dim); config is supplied separately via
            #    the decoded action index.
            context_states = [
                self.env._create_state({}, app.get('stencil_features'))[self.env.config_dim:]
                for app in batch_apps
            ]
            batch_actions = self.agent.act_batch(context_states, training=True,
                                                  verbose_first=True)

            # 3. Build action_configs for synthesis execution.
            action_configs = []
            for i, action in enumerate(batch_actions):
                action_config = self.env.get_config_from_action(action)
                app_info      = batch_apps[i]
                action_configs.append((action, action_config, app_info))

            batch_results = self.env.run_parallel_synthesis_batch(action_configs)
            batch_experiences = []
            for result in batch_results:
                # Store stencil-only state (84-dim).  The pair network reconstructs
                # the full input as cat(stencil, config) using the stored action index.
                state_vector = result['stencil_state']
                action = result['action']
                reward = result['reward']

                # Traceability log: confirm config↔reward pairing entering the buffer.
                cfg  = result['config']
                app_name = result['metrics'].get('app_name', '?') if 'app_name' in result.get('metrics', {}) else '?'
                # app_name is on the job, but result carries config and reward for traceability.
                rw_str = (
                    f"[{', '.join(f'{v:.3f}' for v in reward)}]"
                    if hasattr(reward, '__iter__') and not isinstance(reward, str)
                    else f"{float(reward):.3f}"
                )
                print(
                    f"[Experience] action={action}"
                    f"  p={cfg.get('iter_par_factor','?')} V={cfg.get('vector_factor','?')}"
                    f"  MVF={cfg.get('mem_vector_factor','?')}"
                    f"  rewards={rw_str}"
                )

                batch_experiences.append((state_vector, action, reward))
                self.run_rewards.append(reward[0] if isinstance(reward, np.ndarray) else reward)
            
            self.agent.remember_batch(batch_experiences)
            
            training_losses = []
            if len(self.agent.memory) >= self.agent.batch_size:
                for _ in range(current_batch_size * 2):
                    loss = self.agent.replay()
                    if loss is not None:
                        training_losses.append(loss)
            
            avg_loss = np.mean(training_losses) if training_losses else 0
            self.run_losses.extend([avg_loss] * current_batch_size)
            
            successful_runs = sum(1 for r in batch_results if r['metrics'].get('synthesis_success', False))
            batch_runtime_rewards = [r['reward'][0] if isinstance(r['reward'], np.ndarray) else r['reward'] 
                                    for r in batch_results]
            
            print(f"Success: {successful_runs}/{current_batch_size}, "
                  f"Avg Reward: {np.mean(batch_runtime_rewards):.3f}, "
                  f"Epsilon: {self.agent.epsilon:.3f}")
            
            run += current_batch_size
            if run % save_freq == 0:
                # Check for early stopping
                if len(self.run_rewards) >= reward_window_size:
                    recent_rewards = self.run_rewards[-reward_window_size:]
                    avg_reward = np.mean(recent_rewards)
                    self.agent.update_learning_rate(avg_reward)
                    
                    print(f"Avg reward ({reward_window_size}): {avg_reward:.4f}, LR: {self.agent.optimizer.param_groups[0]['lr']:.2e}")
                    
                    if avg_reward > self.best_avg_reward + early_stopping_min_delta:
                        if self.best_model_path and self.best_model_path.exists():
                            self.best_model_path.unlink()
                        
                        self.best_avg_reward = avg_reward
                        self.patience_counter = 0
                        self.best_model_path = self.model_save_dir / f"dqn_best_{run}.pth"
                        self.agent.save(str(self.best_model_path))
                        print(f"New best model: {self.best_model_path.name} ({avg_reward:.4f})")
                    else:
                        self.patience_counter += 1
                        print(f"No improvement. Patience: {self.patience_counter}/{early_stopping_patience}")
                        
                        if self.patience_counter >= early_stopping_patience:
                            elapsed_hours = (time.time() - start_time) / 3600
                            print(f"\n=== EARLY STOPPING ===")
                            print(f"No improvement for {early_stopping_patience} checks")
                            print(f"Stopped after {run} runs ({elapsed_hours:.1f}h)")
                            print(f"Best model: {self.best_model_path.name} ({self.best_avg_reward:.4f})")
                            break
                
                self.save_results()
        
        print(f"\nTraining complete!")
        if self.best_model_path:
            print(f"Best model: {self.best_model_path.name} ({self.best_avg_reward:.4f})")
        
        return self.run_rewards
    
    def save_results(self):
        """Save training results."""
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        history = getattr(self.env, 'synthesis_history', [])
        serializable_history = [convert_to_serializable(entry) for entry in history]
        with open(self.results_dir / "synthesis_history.json", 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        results = {
            'run_rewards': [float(x) for x in self.run_rewards],
            'run_losses': [float(x) if x is not None else None for x in self.run_losses],
            'total_runs': len(self.run_rewards),
            'synthesis_batch_size': self.synthesis_batch_size,
            'max_parallel_jobs': self.env.max_parallel_jobs
        }
        
        with open(self.results_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.results_dir}")

    # ── Multi-node support ─────────────────────────────────────────────────

    def pretrain_from_stores(self, store_paths: List[str], n_epochs: int = 3) -> int:
        """Offline pre-training on all experiences from one or more existing DBs.

        Loads every row from each DB, shuffles, then cycles through the data
        for *n_epochs* passes — filling the replay buffer in 10 K-row chunks and
        calling agent.replay() after each fill.  No workers or synthesis needed.

        Returns the total number of gradient updates performed.
        """
        from experience_store import ExperienceStore

        # ── Gather all rows ────────────────────────────────────────────────
        all_rows: list = []
        for sp in store_paths:
            store = ExperienceStore(sp)
            rows  = store.pull_all()
            print(f"[Pretrain] {sp}: {len(rows)} experiences")
            all_rows.extend(rows)

        if not all_rows:
            print("[Pretrain] No experiences found — skipping pre-training.")
            return 0

        # App breakdown for diagnostics
        from collections import Counter
        counts = Counter(r['app_name'] for r in all_rows)
        print(f"[Pretrain] Total: {len(all_rows)} experiences across {len(counts)} apps")
        for app, n in sorted(counts.items()):
            print(f"[Pretrain]   {app}: {n}")

        # ── Epoch loop ─────────────────────────────────────────────────────
        import random as _random
        chunk_size    = self.agent.memory.buffer.maxlen  # 10 000
        total_updates = 0

        for epoch in range(1, n_epochs + 1):
            _random.shuffle(all_rows)
            chunk_updates = 0

            for chunk_start in range(0, len(all_rows), chunk_size):
                chunk = all_rows[chunk_start: chunk_start + chunk_size]

                # Fill buffer with this chunk
                for row in chunk:
                    state  = np.asarray(row['stencil_features'], dtype=np.float32)
                    reward = np.asarray(row['reward'],           dtype=np.float32)
                    self.agent.memory.push(
                        Experience(
                            state=state,
                            action=int(row['action']),
                            reward=reward,
                        )
                    )

                # Gradient updates — one pass over the buffer
                if len(self.agent.memory) >= self.agent.batch_size:
                    n_updates = max(1, len(chunk) // self.agent.batch_size)
                    for _ in range(n_updates):
                        loss = self.agent.replay()
                        if loss is not None:
                            self.run_losses.append(loss)
                    chunk_updates += n_updates
                    total_updates += n_updates

            avg_loss = float(np.mean(self.run_losses[-chunk_updates:])) if chunk_updates else 0.0
            print(
                f"[Pretrain] Epoch {epoch}/{n_epochs}  updates={chunk_updates}"
                f"  avg_loss={avg_loss:.4f}  ε={self.agent.epsilon:.3f}"
            )

        print(f"[Pretrain] Done — {total_updates} total gradient updates")
        return total_updates

    def load_experiences_from_store(self, store_path: str) -> int:
        """Pre-fill the replay buffer from a persistent ExperienceStore.

        Safe to call at start-up before training begins, and also mid-training
        to absorb experiences produced by remote worker nodes.

        Returns the number of new experiences loaded.
        """
        from experience_store import ExperienceStore
        store = ExperienceStore(store_path)
        last_id = getattr(self, '_store_last_id', 0)
        rows = store.pull_since(last_id)
        if not rows:
            return 0

        for row in rows:
            state  = np.asarray(row['stencil_features'], dtype=np.float32)
            reward = np.asarray(row['reward'],           dtype=np.float32)
            self.agent.memory.push(
                Experience(
                    state=state,
                    action=int(row['action']),
                    reward=reward,
                )
            )
            self.run_rewards.append(float(reward.mean()) if reward.ndim > 0 else float(reward))

        self._store_last_id = rows[-1]['id']
        print(f"[Store] Loaded {len(rows)} new experiences (buffer now {len(self.agent.memory)})")
        return len(rows)

    def train_from_store(
        self,
        store_path: str,
        shared_checkpoint: str,
        *,
        max_time_hours: float = None,
        save_freq: int = 20,
        poll_interval_s: float = 30.0,
        early_stopping_patience: int = 100,
        early_stopping_min_delta: float = 0.01,
    ) -> List[float]:
        """Trainer-node loop: consume experiences from ExperienceStore and update model.

        This method is intended for use on a CPU-only SLURM node (normal
        partition).  It polls the store for new experiences written by worker
        nodes, performs gradient updates, and saves a shared checkpoint that
        workers reload periodically.

        Parameters
        ----------
        store_path:
            Path to the shared experience_store.db file.
        shared_checkpoint:
            Path where the trainer writes `latest.pth` for workers to reload.
        max_time_hours:
            Wall-clock time budget.  Required if not using early stopping alone.
        save_freq:
            Save checkpoint every this many gradient-update rounds.
        poll_interval_s:
            Seconds to wait between store polls when no new experiences arrive.
        """
        from experience_store import ExperienceStore
        ckpt_path = Path(shared_checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-fill replay buffer from any experiences already in the store
        pre_loaded = self.load_experiences_from_store(store_path)
        print(f"[Trainer] Pre-loaded {pre_loaded} experiences from store")

        start_time  = time.time()
        max_time_s  = max_time_hours * 3600 if max_time_hours else None
        reward_window_size = 20
        update_round = 0

        print(f"[Trainer] Starting store-driven training loop (poll every {poll_interval_s}s)")

        while True:
            if max_time_s and (time.time() - start_time) >= max_time_s:
                print(f"\n[Trainer] Time budget reached")
                break

            # Pull new experiences from workers
            new_count = self.load_experiences_from_store(store_path)

            if len(self.agent.memory) < self.agent.batch_size:
                print(f"[Trainer] Buffer too small ({len(self.agent.memory)}/{self.agent.batch_size}), waiting…")
                time.sleep(poll_interval_s)
                continue

            # Perform gradient updates — more updates when more new data arrives
            n_updates = max(1, new_count * 2)
            losses = []
            for _ in range(n_updates):
                loss = self.agent.replay()
                if loss is not None:
                    losses.append(loss)

            avg_loss = np.mean(losses) if losses else 0.0
            self.run_losses.append(avg_loss)
            update_round += 1

            elapsed_h = (time.time() - start_time) / 3600
            print(
                f"[Trainer] round={update_round}  new={new_count}  updates={n_updates}"
                f"  loss={avg_loss:.4f}  ε={self.agent.epsilon:.3f}"
                f"  buffer={len(self.agent.memory)}  elapsed={elapsed_h:.2f}h"
            )

            # Save shared checkpoint every save_freq rounds
            if update_round % save_freq == 0:
                # Always overwrite latest.pth so workers pick up the new policy
                self.agent.save(str(ckpt_path))
                print(f"[Trainer] Saved shared checkpoint → {ckpt_path}")

                # Best-model tracking
                if len(self.run_rewards) >= reward_window_size:
                    recent  = self.run_rewards[-reward_window_size:]
                    avg_rew = float(np.mean(recent))
                    self.agent.update_learning_rate(avg_rew)

                    if avg_rew > self.best_avg_reward + early_stopping_min_delta:
                        if self.best_model_path and Path(self.best_model_path).exists():
                            Path(self.best_model_path).unlink()
                        self.best_avg_reward = avg_rew
                        self.patience_counter = 0
                        self.best_model_path = str(self.model_save_dir / f"dqn_best_{update_round}.pth")
                        self.agent.save(self.best_model_path)
                        print(f"[Trainer] New best model: dqn_best_{update_round}.pth ({avg_rew:.4f})")
                    else:
                        self.patience_counter += 1
                        print(f"[Trainer] No improvement. Patience: {self.patience_counter}/{early_stopping_patience}")
                        if self.patience_counter >= early_stopping_patience:
                            print(f"[Trainer] Early stopping triggered")
                            break

                self.save_results()

            # If no new experiences, pause before next poll
            if new_count == 0:
                time.sleep(poll_interval_s)

        # Final checkpoint save
        self.agent.save(str(ckpt_path))
        print(f"\n[Trainer] Training complete. Best reward: {self.best_avg_reward:.4f}")
        return self.run_rewards

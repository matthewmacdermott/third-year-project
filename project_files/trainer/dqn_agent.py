"""DQN Agent for multi-objective reinforcement learning.

Uses a pairwise (action-as-input) Q-network: Q(stencil, config) → R^6.
The agent samples K config indices at inference, builds a (K, 6) Q-matrix,
and selects from the Pareto front.  No giant weight matrix is needed.
"""

import random
import threading
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, List, Optional, Tuple

from dqn_network import DQNNetwork
from experience import Experience, ReplayBuffer


class DQNAgent:
    """Multi-Objective DQN agent for HLS optimisation (pair-network)."""

    # Objective names — must match _calculate_reward order in hls_environment.py.
    OBJECTIVE_NAMES = ['runtime', 'bram', 'dsp', 'lut', 'ff', 'freq']

    def __init__(self, stencil_dim: int, config_dim: int, action_dim: int,
                 decode_action_fn: Callable[[int], np.ndarray],
                 is_valid_fn: Callable[[int], bool] = None,
                 learning_rate: float = 1e-3, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 100,
                 device: str = None, num_objectives: int = 6):
        """
        Args:
            stencil_dim:      Dimensionality of stencil-feature state (e.g. 84).
            config_dim:       Dimensionality of normalised config vector (e.g. 10).
            action_dim:       Total number of discrete actions (e.g. 9_175_040).
            decode_action_fn: Callable(action_idx) → np.ndarray(config_dim,) —
                              converts an integer action index to a normalised
                              config vector.  env.get_config_vector() satisfies this.
            is_valid_fn:      Optional callable(action_idx) → bool.  When provided,
                              both random exploration and exploit sampling skip
                              actions that return False, so structurally invalid
                              configs (e.g. MVF < VF) never enter the replay buffer.
        """
        self.stencil_dim = stencil_dim
        self.config_dim = config_dim
        self.action_dim = action_dim
        self.decode_action_fn = decode_action_fn
        self.is_valid_fn = is_valid_fn
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.num_objectives = num_objectives

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.q_network = DQNNetwork(
            stencil_dim, config_dim, hidden_dims=[256, 128, 64], num_objectives=num_objectives
        ).to(self.device)
        self.target_network = DQNNetwork(
            stencil_dim, config_dim, hidden_dims=[256, 128, 64], num_objectives=num_objectives
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-6
        )

        self.memory = ReplayBuffer(memory_size)
        self.training_step = 0
        self._thread_local = threading.local()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_thread_random(self) -> random.Random:
        """Return a thread-local Random instance."""
        if not hasattr(self._thread_local, 'rng'):
            tid = threading.get_ident()
            seed = int((tid * 1_000_000 + int(time.time() * 1_000_000)) % (2**32))
            self._thread_local.rng = random.Random(seed)
        return self._thread_local.rng

    def _decode_actions_batch(self, action_indices) -> np.ndarray:
        """Decode a sequence of action indices to a (N, config_dim) float32 array."""
        return np.array(
            [self.decode_action_fn(int(a)) for a in action_indices],
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Pareto selection (vectorised, operates on numpy arrays)
    # ------------------------------------------------------------------

    def _get_pareto_optimal_actions(self, q_values: np.ndarray) -> List[int]:
        """Find Pareto-optimal column indices in q_values.

        q_values: (num_objectives, num_actions) — higher is better for all.
        Returns a list of column indices that are not dominated.
        """
        q = q_values.T        # (n, k)
        n = q.shape[0]
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            diff = q[i] - q   # (n, k)
            j_dominates_i = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
            j_dominates_i[i] = False
            if np.any(j_dominates_i):
                is_dominated[i] = True
                continue
            i_dominates_j = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
            i_dominates_j[i] = False
            is_dominated[i_dominates_j] = True
        return list(np.where(~is_dominated)[0])

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, stencil_state: np.ndarray, training: bool = True,
            num_action_samples: int = 2000, verbose: bool = False) -> int:
        """Choose an action using epsilon-greedy + Pareto selection.

        Args:
            stencil_state:      Stencil feature vector (stencil_dim,).
            training:           If True, epsilon-greedy exploration is active.
            num_action_samples: Number of configs to sample during exploitation.
            verbose:            Print Q-value diagnostics for this call.
        """
        if training and self._get_thread_random().random() < self.epsilon:
            if self.is_valid_fn is not None:
                # Re-sample until we find a valid action (max 200 attempts).
                for _ in range(200):
                    chosen = self._get_thread_random().randint(0, self.action_dim - 1)
                    if self.is_valid_fn(chosen):
                        break
            else:
                chosen = self._get_thread_random().randint(0, self.action_dim - 1)
            if verbose:
                print(
                    f"[DQN] EXPLORE  ε={self.epsilon:.4f}  "
                    f"random action={chosen}  (action_dim={self.action_dim})"
                )
            return chosen

        with torch.no_grad():
            # Limit sampling to avoid evaluating all 6.8M configurations.
            # Instead, sample a manageable subset and find the best trade-off.
            num_samples = min(num_action_samples, self.action_dim)
            sampled_actions = self._get_thread_random().sample(
                range(self.action_dim), num_samples
            )
            
            # Drop structurally invalid actions so they never influence Q-values.
            # E.g., configs where mem_vector_factor < vector_factor are nonsensical
            # and would waste synthesis time. This legality check prevents them from
            # entering the decision-making pipeline.
            if self.is_valid_fn is not None:
                sampled_actions = [a for a in sampled_actions if self.is_valid_fn(a)]
                if not sampled_actions:
                    # Extremely unlikely; fall back to a fresh random valid action.
                    return self.act(stencil_state, training=False,
                                   num_action_samples=num_action_samples, verbose=verbose)

            # Build (K, stencil_dim + config_dim) input batch — tile AFTER filtering
            # so stencil_tiled rows match the (possibly reduced) sampled_actions list.
            # The network expects concatenated [stencil, config] pairs, one per action.
            config_matrix = self._decode_actions_batch(sampled_actions)  # (K, config_dim)
            stencil_tiled = np.tile(stencil_state, (len(sampled_actions), 1))  # (K, stencil_dim)
            inputs = np.concatenate([stencil_tiled, config_matrix], axis=1)  # (K, input_dim=94)

            # Forward pass: evaluate all K (state, action) pairs in parallel.
            # Returns shape (K, num_objectives), one Q-value vector per config.
            inputs_t = torch.FloatTensor(inputs).to(self.device)
            q_out = self.q_network(inputs_t).cpu().numpy()   # (K, num_objectives)
            multi_q_values = q_out.T                          # (num_objectives, K)

            if verbose:
                print(
                    f"[DQN] EXPLOIT  ε={self.epsilon:.4f}  "
                    f"sampled {num_samples}/{self.action_dim} actions  "
                    f"q_matrix shape={multi_q_values.shape} "
                    f"(objectives x sampled_actions)"
                )
                for obj_idx, obj_name in enumerate(self.OBJECTIVE_NAMES):
                    q_row = multi_q_values[obj_idx]
                    print(
                        f"[DQN]   obj={obj_name:<8s}  "
                        f"min={q_row.min():+.4f}  max={q_row.max():+.4f}  "
                        f"mean={q_row.mean():+.4f}"
                    )

            pareto_actions = self._get_pareto_optimal_actions(multi_q_values)

            if not pareto_actions:
                best_idx = multi_q_values[0].argmax()
                chosen = sampled_actions[best_idx]
                if verbose:
                    print(
                        f"[DQN] Pareto front empty — fallback greedy on obj[0]  "
                        f"chosen action={chosen}"
                    )
                return chosen

            pareto_idx = self._get_thread_random().choice(pareto_actions)
            chosen = sampled_actions[pareto_idx]

            if verbose:
                print(
                    f"[DQN] Pareto front: {len(pareto_actions)}/{num_samples} actions  "
                    f"→ randomly selected sample_idx={pareto_idx}  action={chosen}"
                )
                q_chosen = multi_q_values[:, pareto_idx]
                q_parts = "  ".join(
                    f"{n}={v:+.4f}" for n, v in zip(self.OBJECTIVE_NAMES, q_chosen)
                )
                print(f"[DQN] Q-values for chosen action:  {q_parts}")

            return chosen

    def act_batch(self, stencil_states: List[np.ndarray], training: bool = True,
                  verbose: bool = False, verbose_first: bool = False) -> List[int]:
        """Choose actions for a batch of stencil states."""
        return [
            self.act(
                state, training,
                verbose=(verbose or (verbose_first and i == 0))
            )
            for i, state in enumerate(stencil_states)
        ]

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def remember_batch(self, experiences: List[Tuple]) -> None:
        """Store a batch of (state, action, reward) tuples."""
        for state, action, reward in experiences:
            self.memory.push(Experience(state, action, reward))

    def replay(self) -> Optional[float]:
        """Sample replay memory and run one DQN optimisation step.

        Returns the scalar training loss, or None when the replay buffer does
        not yet contain enough samples for a full mini-batch.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(np.array([e.state      for e in batch])).to(self.device)
        actions_np  = np.array([e.action for e in batch])
        rewards     = torch.FloatTensor(np.array([e.reward     for e in batch])).to(self.device)

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1).expand(-1, self.num_objectives)

        # ---- Current Q-values ----
        # Decode actions → config vectors, cat with stencil states
        config_matrix = torch.FloatTensor(
            self._decode_actions_batch(actions_np)
        ).to(self.device)                                    # (batch, config_dim)
        current_inputs = torch.cat([states, config_matrix], dim=1)  # (batch, input_dim)
        current_q = self.q_network(current_inputs)           # (batch, num_objectives)

        # ---- Target Q-values ----
        # One-step contextual update: regress directly to observed reward.
        with torch.no_grad():
            target_q = rewards

        # Regression objective: fit observed objective values for each (state, action).
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid unstable spikes from rare large updates.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()

        self.training_step += 1
        # Periodically copy online weights into the target network. Keeping the
        # target fixed between syncs stabilises Bellman targets.
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Gradually reduce exploration pressure over training.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def batch_q_values(self, stencil_state: np.ndarray, config_matrix: np.ndarray,
                       batch_size: int = 65536) -> np.ndarray:
        """Evaluate (N, config_dim) normalised configs against stencil_state.

        Processes in chunks so memory usage stays bounded regardless of N.
        """
        N = len(config_matrix)
        q_out = np.empty((N, self.num_objectives), dtype=np.float32)
        stencil_t = torch.FloatTensor(stencil_state).to(self.device)
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                cfg_t = torch.FloatTensor(config_matrix[start:end]).to(self.device)
                s_t = stencil_t.unsqueeze(0).expand(end - start, -1)
                inp = torch.cat([s_t, cfg_t], dim=1)
                q_out[start:end] = self.q_network(inp).cpu().numpy()
        return q_out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def update_learning_rate(self, metric: float) -> None:
        self.scheduler.step(metric)

    def save(self, filepath: str) -> None:
        torch.save({
            'stencil_dim':                self.stencil_dim,
            'config_dim':                 self.config_dim,
            'q_network_state_dict':       self.q_network.state_dict(),
            'target_network_state_dict':  self.target_network.state_dict(),
            'optimizer_state_dict':       self.optimizer.state_dict(),
            'scheduler_state_dict':       self.scheduler.state_dict(),
            'epsilon':                    self.epsilon,
            'training_step':              self.training_step,
        }, filepath)

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon       = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']

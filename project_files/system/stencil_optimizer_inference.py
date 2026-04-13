#!/usr/bin/env python3
"""Inference tool for HLS configuration recommendations using trained DQN model."""

import argparse
import heapq
import json
import numpy as np
import sys
import torch
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / 'trainer'))
from dqn_agent import DQNAgent
from hls_environment import ParallelHLSEnvironment
from config_space import action_space_size, decode_action_to_config, normalise_config
from stencil_feature_loader import StencilFeatureLoader


class StencilOptimizer:

    def __init__(self, model_path: str):
        print("=== HLS Configuration Optimizer ===\n")

        config_path = Path(__file__).parent.parent / 'hls_param_ranges.json'
        with open(config_path, 'r') as f:
            self.param_ranges = json.load(f)

        self.action_dim = action_space_size(self.param_ranges)

        self.stencil_loader = StencilFeatureLoader()

        self.config_dim = 10
        self.max_stencil_points = 27
        self.stencil_dim = 3 + (self.max_stencil_points * 3)

        self.agent = DQNAgent(
            stencil_dim=self.stencil_dim,
            config_dim=self.config_dim,
            action_dim=self.action_dim,
            decode_action_fn=self._get_config_vector,
            is_valid_fn=lambda idx: ParallelHLSEnvironment._is_config_valid(
                self.get_config_from_action(idx), param_ranges=self.param_ranges
            )[0],
            learning_rate=1e-3,
            epsilon=0.0
        )
        if Path(model_path).exists():
            self.agent.load(model_path)
            self.agent.q_network.eval()
            print(f"Loaded model: {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
            self.agent.q_network.eval()

        print(f"Action space: {self.action_dim} configurations")

    def _get_config_vector(self, action_idx: int) -> np.ndarray:
        return normalise_config(self.get_config_from_action(action_idx), self.param_ranges)

    def score_config(self, config: Dict, state: np.ndarray, strategy: str = 'balanced') -> float:
        """Return the strategy score for an arbitrary config dict against a stencil state vector."""
        v = normalise_config(config, self.param_ranges)
        q = self.agent.batch_q_values(state, v[np.newaxis, :])[0]
        weights = self._strategy_weights(strategy)
        return float((q * weights).sum())

    def get_config_from_action(self, action_idx: int) -> Dict:
        """Decode action index to configuration."""
        return decode_action_to_config(action_idx, self.param_ranges)
    
    def optimize_project(self, project_dir: str, top_k: int = 5, strategy: str = 'balanced'):
        """Get top-k configuration recommendations.
        
        Args:
            project_dir: Path to HLS project
            top_k: Number of recommendations to return
            strategy: Selection strategy - 'balanced', 'resource', or 'performance'
        """
        project_path = Path(project_dir)
        print(f"Optimizing: {project_path.name}")
        
        stencil_features = self.stencil_loader.load_from_project(project_path)
        if stencil_features:
            app_name = stencil_features.get('file_info', {}).get('app_name', project_path.parent.name)
            print(f"Application: {app_name}")
            num_stencils = stencil_features.get('total_stencils', 0)
            print(f"Stencils: {num_stencils}\n")
        else:
            print("No stencil features found\n")
        
        state = np.zeros(self.stencil_dim, dtype=np.float32)
        if stencil_features:
            stencil_vector = self.stencil_loader.extract_features_vector(
                stencil_features, self.max_stencil_points
            )
            state[:] = stencil_vector
            print(f"Stencil vector shape: {state.shape}, non-zero elements: {np.count_nonzero(state)}")
        
        print(f"\nStrategy: {strategy}")
        print(f"Exhaustive search over all {self.action_dim} actions...")

        top_actions = self._get_top_k_exhaustive(state, strategy, top_k)
        if not top_actions:
            raise RuntimeError(
                "Exhaustive search returned no valid recommendations. "
                "Check model validity filter or action-space constraints."
            )

        recommendations = []
        for rank, item in enumerate(top_actions, start=1):
            q_vals = item['q_values']
            recommendations.append({
                'rank': rank,
                'config': item['config'],
                'q_values': [q_vals['latency'], q_vals['bram'], q_vals['dsp'],
                             q_vals['lut'], q_vals['ff'], q_vals['frequency']],
                'stencil_context': {
                    'app_name': stencil_features.get('file_info', {}).get('app_name') if stencil_features else None,
                    'total_stencils': stencil_features.get('total_stencils', 0) if stencil_features else 0
                },
                'strategy_score': item['strategy_score']
            })
        
        # Display recommendations
        print("Configuration Recommendations:")
        print("=" * 60)
        objective_names = ['latency', 'bram', 'dsp', 'lut', 'ff', 'frequency']
        for rec in recommendations:
            print(f"\nRank {rec['rank']}:")
            # Display Q-values for each objective
            q_vals = rec['q_values'] if isinstance(rec['q_values'], list) else [rec['q_values']]
            if len(q_vals) > 1:
                print("  Objective Q-values:")
                for i, (obj_name, q_val) in enumerate(zip(objective_names, q_vals)):
                    print(f"    {obj_name:10s}: {q_val:8.4f}")
            else:
                print(f"  Q-value: {q_vals[0]:.4f}")
                
            if rec['stencil_context']['app_name']:
                print(f"  App: {rec['stencil_context']['app_name']} ({rec['stencil_context']['total_stencils']} stencils)")
            print("  Configuration:")
            for param, value in rec['config'].items():
                print(f"    \"{param}\" : {value},")
        
        return recommendations
    
    # ------------------------------------------------------------------
    # Exhaustive search helpers
    # ------------------------------------------------------------------

    def _strategy_weights(self, strategy: str) -> np.ndarray:
        """Return 6-dim weight vector [latency, bram, dsp, lut, ff, frequency]."""
        if strategy == 'performance':
            # Only runtime and clock frequency matter
            return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
        elif strategy == 'resource':
            # Minimise resource utilisation
            return np.array([0.0, 0.25, 0.25, 0.25, 0.25, 0.0], dtype=np.float32)
        else:  # balanced
            return np.array([0.25, 0.125, 0.125, 0.125, 0.125, 0.25], dtype=np.float32)

    def _decode_configs_vectorized(self, start: int, count: int) -> np.ndarray:
        """Vectorized decode of action indices [start, start+count)."""
        params = list(self.param_ranges.items())
        config_matrix = np.empty((count, len(params)), dtype=np.float32)
        remainder = np.arange(start, start + count, dtype=np.int64)
        for col, (name, values) in reversed(list(enumerate(params))):
            n = len(values)
            param_vals = np.array(values, dtype=np.float32)[remainder % n]
            low = float(min(values))
            scale = float(max(values) - low)
            config_matrix[:, col] = np.clip((param_vals - low) / scale, 0.0, 1.0)
            remainder //= n
        return config_matrix

    def _get_top_k_exhaustive(self, state: np.ndarray, strategy: str, top_k: int,
                               batch_size: int = 65536) -> List[dict]:
        """Score every action in the full action space and return top-k.

        Uses a vectorized numpy decoder and batched Q-network forward passes so
        the entire 6.88 M-action space is covered in ~10-20 s on CPU.
        A min-heap of size POOL_SIZE tracks the best candidates across batches;
        validity filtering (is_valid_fn) is applied only to the small pool.
        """
        POOL_SIZE = max(top_k * 40, 200)  # candidates before validity filtering
        weights = self._strategy_weights(strategy)
        obj_names = ['latency', 'bram', 'dsp', 'lut', 'ff', 'frequency']

        # Min-heap of (score, action_idx) — keeps the POOL_SIZE highest *valid* scores
        heap: list = []
        check_validity = self.agent.is_valid_fn is not None

        n_batches = (self.action_dim + batch_size - 1) // batch_size
        print(f"  {n_batches} batches × {batch_size} actions")

        for b_start in range(0, self.action_dim, batch_size):
            b_end = min(b_start + batch_size, self.action_dim)
            count = b_end - b_start
            cfg_mat = self._decode_configs_vectorized(b_start, count)      # (count, 10)
            q_vals  = self.agent.batch_q_values(state, cfg_mat)            # (count, 6)
            scores  = (q_vals * weights).sum(axis=1)                       # (count,)

            # Keep top-50 from this batch to avoid per-element heap ops
            k_local = min(50, count)
            top_local = np.argpartition(scores, -k_local)[-k_local:]
            for li in top_local:
                aidx = int(b_start + li)
                # Validity-check here so invalid configs never occupy pool slots
                if check_validity and not self.agent.is_valid_fn(aidx):
                    continue
                heapq.heappush(heap, (float(scores[li]), aidx))
                if len(heap) > POOL_SIZE:
                    heapq.heappop(heap)  # removes smallest — keeps top-POOL_SIZE

        # Sort candidates best-first
        candidates = sorted(heap, key=lambda x: -x[0])

        # Build final top-k with per-action Q-value lookup (all candidates already valid)
        results: List[dict] = []
        for score, aidx in candidates:
            cfg_vec = self._get_config_vector(aidx)
            q_single = self.agent.batch_q_values(state, cfg_vec[np.newaxis, :])  # (1, 6)
            results.append({
                'action':         aidx,
                'config':         self.get_config_from_action(aidx),
                'strategy_score': score,
                'q_values':       {n: float(q_single[0, i]) for i, n in enumerate(obj_names)},
            })
            if len(results) == top_k:
                break
        return results

    def _rank_by_strategy(self, pareto_front: list, strategy: str) -> list:
        """Rank Pareto optimal actions based on strategy.
        
        Args:
            pareto_front: List of Pareto optimal actions with Q-values
            strategy: 'balanced', 'resource', or 'performance'
        
        Returns:
            Sorted list of actions with strategy scores
        """
        scored_actions = []
        
        for action_data in pareto_front:
            q_vals = action_data['q_values']
            
            if strategy == 'performance':
                # All reward signals are higher=better:
                # latency_reward = 1 - runtime/100000  (higher → faster)
                # freq_reward    = freq/300            (higher → faster clock)
                score = q_vals['latency'] + q_vals['frequency']
                
            elif strategy == 'resource':
                # bram/dsp/lut/ff rewards = 1 - utilisation (higher → less used)
                score = q_vals['bram'] + q_vals['dsp'] + q_vals['lut'] + q_vals['ff']
                
            else:  # balanced
                perf_score = q_vals['latency'] + q_vals['frequency']
                resource_score = q_vals['bram'] + q_vals['dsp'] + q_vals['lut'] + q_vals['ff']
                score = 0.5 * perf_score + 0.5 * resource_score
            
            action_data['strategy_score'] = score
            scored_actions.append(action_data)
        
        # Sort by score (descending - higher is better)
        scored_actions.sort(key=lambda x: x['strategy_score'], reverse=True)
        return scored_actions
    
    def export_config(self, config: Dict, output_file: str):
        """Export configuration to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4, separators=(',', ' : '))
        print(f"\nConfiguration exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='HLS Configuration Optimization with DQN')
    parser.add_argument('--strategy', type=str, default='balanced', 
                        choices=['balanced', 'resource', 'performance'],
                        help='Selection strategy: balanced (default), resource (minimize usage), or performance (maximize speed)')
    parser.add_argument('project', type=str, help='Path to project directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained PyTorch model')
    parser.add_argument('--top-k', type=int, default=5, help='Number of recommendations')
    parser.add_argument('--export', type=str, help='Export best config to JSON file')
    args = parser.parse_args()
    
    # Auto-detect project directory
    project_path = Path(args.project)
    if project_path.is_file():
        project_path = project_path.parent
        for _ in range(3):
            if (project_path / 'Makefile').exists():
                break
            project_path = project_path.parent
    
    print(f"Project directory: {project_path}\n")
    optimizer = StencilOptimizer(model_path=args.model)
    recommendations = optimizer.optimize_project(str(project_path), args.top_k, args.strategy)
    
    if args.export and recommendations:
        optimizer.export_config(recommendations[0]['config'], args.export)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

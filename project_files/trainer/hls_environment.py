"""Parallel HLS environment for RL optimization."""

import json
import math
import multiprocessing
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import shutil

from synthesis_runner import run_synthesis_with_progress
from config_space import action_space_size, decode_action_to_config, normalise_config
from fpga_specs import U280_MAXI_WIDTH_BITS


def _is_infrastructure_error(msg: str) -> bool:
    """Return True if *msg* describes an environment/infrastructure failure.

    Such failures (missing tool, disk full, network error) should not be
    treated as config validity failures, because they say nothing about
    whether the config itself is valid.
    """
    if not msg:
        return False
    lower = msg.lower()
    return any(kw in lower for kw in (
        'no such file or directory',
        'filenotfounderror',
        'permissionerror',
        'oserror',
        'disk quota',
        'no space left',
        'broken pipe',
        'connection refused',
        'timeout',
        'killed',
        'signal',
        'vitis_hls not found',
        'command not found',
    ))
from stencil_feature_loader import StencilFeatureLoader


class ParallelHLSEnvironment:
    """HLS environment for RL optimization."""

    def __init__(self, apps_base_dir: str = "codegen_apps",
                 max_parallel_jobs: int = None,
                 enable_stencil_features: bool = True):
        self.apps_base_dir = Path(apps_base_dir)
        if not self.apps_base_dir.is_absolute():
            self.apps_base_dir = Path.cwd() / self.apps_base_dir
            
        self.enable_stencil_features = enable_stencil_features
        self.stencil_loader = StencilFeatureLoader() if enable_stencil_features else None
        
        cpu_count = multiprocessing.cpu_count()
        if max_parallel_jobs is None:
            # vitis_hls uses ~4 threads internally; cap at cpu_count//4 so we
            # don't saturate the node, and never exceed 32.
            self.max_parallel_jobs = max(1, min(cpu_count // 4, 32))
        else:
            self.max_parallel_jobs = max(1, min(max_parallel_jobs, cpu_count))

        config_path = Path(__file__).parent.parent / 'hls_param_ranges.json'
        with open(config_path, 'r') as f:
            self.param_ranges = json.load(f)
        self.action_dim = action_space_size(self.param_ranges)
        
        # Use FPGA platform specifications from fpga_specs module
        self.u280_maxi_width_bits = U280_MAXI_WIDTH_BITS
        
        self.config_dim = 10
        self.max_stencil_points = 27
        self.stencil_dim = 3 + (self.max_stencil_points * 3)
        self.state_dim = self.config_dim + self.stencil_dim
        
        self.available_apps = self._find_available_apps()
        self._weighted_apps = self._build_weighted_app_list()
        # Cache dynamically inferred benchmark sizes:
        # app_name -> (n_iter, nx, ny, nz)
        self._benchmark_size_cache: Dict[str, Tuple[int, int, int, int]] = {}
        # Lock protecting shared mutable training state across synthesis threads.
        self._state_lock = threading.Lock()
        # Keep a per-run log of synthesis outcomes used by result export.
        self.synthesis_history = []
    
    def get_config_from_action(self, action_idx: int) -> Dict:
        """Convert action index to configuration dictionary."""
        if action_idx < 0 or action_idx >= self.action_dim:
            raise ValueError(f"Action index {action_idx} out of range [0, {self.action_dim})")

        return decode_action_to_config(action_idx, self.param_ranges)

    def get_config_vector(self, action_idx: int) -> np.ndarray:
        """Return the normalised config vector for an action index."""
        return normalise_config(self.get_config_from_action(action_idx), self.param_ranges)

    @staticmethod
    def _resource_reward(util: float, safe_threshold: float = 0.80) -> float:
        """Soft-cliff resource reward.

        Returns 1.0 for utilisation up to *safe_threshold*, linearly drops to
        0.0 at 100% utilisation, and is 0.0 beyond (overflow).

        The safe zone avoids penalising valid high-utilisation configs; the
        linear ramp in (safe_threshold, 1.0] gives the network a gradient
        signal when approaching the overflow boundary.
        """
        if util <= safe_threshold:
            return 1.0
        elif util <= 1.0:
            return (1.0 - util) / (1.0 - safe_threshold)
        else:
            return 0.0

    @staticmethod
    def _util_tag(util: float, safe_threshold: float = 0.80) -> str:
        """Short human-readable label for a utilisation value in the synth log."""
        if util > 1.0:
            return 'OVERFLOW'
        elif util > safe_threshold:
            return 'WARNING'
        else:
            return 'ok'

    @staticmethod
    def _is_config_valid(config: Dict,
                        platform_maxi_width_bits: int = 512,
                        param_ranges: Optional[Dict] = None) -> Tuple[bool, str]:
        """Pre-validate a config against known U280 hardware constraints.

        Checks performed (cheap, no synthesis required):

        1. mem_vector_factor >= vector_factor

        2. mem_vector_factor * 32 <= platform_maxi_width_bits

        3. maxi_depth >= maxi_read_burst_length * num_read_outstanding and
           maxi_depth >= maxi_write_burst_length * num_write_outstanding

        Returns (True, '') when valid, (False, reason) otherwise.
        """
        # Check every param value is in the allowed list.
        if param_ranges:
            for param, allowed in param_ranges.items():
                val = config.get(param)
                if val is not None and val not in allowed:
                    return False, (
                        f"{param}={val} is not a valid option "
                        f"(allowed: {allowed})"
                    )

        vf  = config['vector_factor']
        mvf = config['mem_vector_factor']
        if mvf < vf:
            return False, f"mem_vector_factor={mvf} < vector_factor={vf}"

        maxi_port_bits = mvf * 32
        if maxi_port_bits > platform_maxi_width_bits:
            return False, (
                f"mem_vector_factor={mvf} → AXI port width {maxi_port_bits} bits "
                f"exceeds platform max {platform_maxi_width_bits} bits"
            )

        md  = config['maxi_depth']
        brl = config['maxi_read_burst_length']
        bwl = config['maxi_write_burst_length']
        nro = config['num_read_outstanding']
        nwo = config['num_write_outstanding']
        if md < brl * nro:
            return False, (f"maxi_depth={md} < "
                           f"maxi_read_burst_length({brl}) * "
                           f"num_read_outstanding({nro}) = {brl*nro}")
        if md < bwl * nwo:
            return False, (f"maxi_depth={md} < "
                           f"maxi_write_burst_length({bwl}) * "
                           f"num_write_outstanding({nwo}) = {bwl*nwo}")

        return True, ''
    
    def _find_available_apps(self) -> List[Dict]:
        """Find all available HLS applications."""
        available_apps = []

        if not self.apps_base_dir.exists():
            print(f"Warning: Apps directory not found: {self.apps_base_dir}")
            return available_apps
        
        for app_dir in self.apps_base_dir.iterdir():
            if not app_dir.is_dir():
                continue
            
            for project_dir in app_dir.glob("*_project"):
                if "u280" not in project_dir.name.lower():
                    continue
                
                config_files = list(project_dir.glob("config_*.json"))
                makefile = project_dir / "Makefile"
                
                if config_files and makefile.exists():
                    stencil_features = None
                    if self.enable_stencil_features and self.stencil_loader:
                        stencil_features = self.stencil_loader.load_from_project(project_dir)
                        if stencil_features:
                            dims = stencil_features.get('program_dimensions', '?')
                            stencils = stencil_features.get('stencils', [])
                            summary = ', '.join(
                                f"{s.get('name', '?')}(pts={s.get('num_points', '?')},r={s.get('max_radius', '?')})"
                                for s in stencils
                            )
                            print(
                                f"[StencilParse] {app_dir.name}/{project_dir.name}: "
                                f"dims={dims} stencils={len(stencils)} [{summary}]"
                            )
                    
                    available_apps.append({
                        'app_name': app_dir.name,
                        'project_name': project_dir.name,
                        'project_dir': project_dir,
                        'config_file': config_files[0],
                        'stencil_features': stencil_features
                    })
        
        return available_apps

    def _build_weighted_app_list(self) -> List[Dict]:
        """Return a list where 3-D apps appear 3x as often as 2-D apps.

        Implemented as a flat repeated list so `random.choice` gives the
        correct probabilities without needing explicit weights at sample time.
        """
        
        _3D_WEIGHT = 3
        weighted: List[Dict] = []
        for app in self.available_apps:
            sf = app.get('stencil_features') or {}
            dims = sf.get('file_info', {}).get('program_dimensions')
            # Skip apps with no valid dimension
            if dims is None:
                continue
            repeats = _3D_WEIGHT if dims >= 3 else 1
            weighted.extend([app] * repeats)
        return weighted

    def sample_batch_apps(self, batch_size: int) -> List[Dict]:
        """Sample a list of app_info dicts for a batch, one per job.

        3-D stencils are sampled 3x more often than 2-D stencils to compensate
        for the smaller number of 3-D apps in the training set.
        """
        if not self.available_apps:
            raise RuntimeError("No available apps found!")
        return [random.choice(self._weighted_apps) for _ in range(batch_size)]
    
    def _create_state(self, config: Dict, stencil_features: Optional[Dict] = None) -> np.ndarray:
        """Create state representation from configuration and stencil features.

        Args:
            config: HLS parameter dictionary.
            stencil_features: Parsed stencil feature dict from StencilFeatureLoader.
        """
        state_vector = np.zeros(self.state_dim, dtype=np.float32)
        
        state_vector[0:10] = normalise_config(config, self.param_ranges)

        stencil_vector = self.stencil_loader.extract_features_vector(stencil_features, self.max_stencil_points)
        state_vector[10:] = stencil_vector
        
        return state_vector
    
    def run_parallel_synthesis_batch(self, action_configs: List[Tuple]) -> List[Dict]:
        """Run a batch of synthesis jobs in parallel.

        Each element of *action_configs* must be a 3-tuple:
            (action_index, config_dict, app_info_dict)
        The app_info must be the same one whose stencil features were used to
        build the state shown to the agent, so the network sees consistent
        (state, reward) pairs.
        """
        if not action_configs:
            return []
        
        print(f"Running {len(action_configs)} synthesis jobs...")
        
        # Prepare jobs — use the caller-supplied app_info, NOT a new random one
        jobs = []
        for action, config, app_info in action_configs:
            jobs.append({
                'action': action,
                'config': config,
                'app_info': app_info
            })
        
        # Execute jobs in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
            future_to_job = {executor.submit(self._run_synthesis_job, job): job for job in jobs}
            
            for future in as_completed(future_to_job):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    job = future_to_job[future]
                    print(f"Job failed: {exc}")
                    results.append({
                        'app_name': job['app_info']['app_name'],
                        'action': job['action'],
                        'config': job['config'],
                        'metrics': {'synthesis_success': False},
                        'reward': np.array([-1.0] * 6, dtype=np.float32),  # Multi-objective failure penalty
                        'stencil_state': self._create_state({}, job['app_info'].get('stencil_features'))[self.config_dim:]
                    })
        
        successful = sum(1 for r in results if r['metrics'].get('synthesis_success', False))
        print(f"Completed: {successful}/{len(results)} successful")
        
        return results
    
    def _run_synthesis_job(self, job_data: Dict) -> Dict:
        """Run a single synthesis job."""
        app_info = job_data['app_info']
        config = job_data['config']
        action = job_data['action']
        
        metrics = self._run_synthesis(app_info, config)
        reward = self._calculate_reward(metrics, config, app_info)
        stencil_state = self._create_state({}, app_info.get('stencil_features'))[self.config_dim:]

        # Track synthesis history — lock because multiple threads write concurrently.
        history_entry = {
            'app_name': app_info['app_name'],
            'config': config.copy(),
            'metrics': metrics.copy(),
            'reward': reward
        }
        with self._state_lock:
            self.synthesis_history.append(history_entry)
        
        return {
            'app_name': app_info['app_name'],
            'action': action,
            'config': config,
            'metrics': metrics,
            'reward': reward,
            'stencil_state': stencil_state
        }
    
    def _run_synthesis(self, app_info: Dict, config: Dict) -> Dict:
        """Run HLS synthesis and return metrics."""
        app_name = app_info['app_name']

        # Pre-validate hardware constraints before spending any synthesis time.
        valid, reason = self._is_config_valid(config, self.u280_maxi_width_bits, self.param_ranges)
        if not valid:
            print(f"[HLSEnv] Pre-validation rejected config for {app_name}: {reason}")
            return {
                'synthesis_success': False,
                'error_message': f'Invalid config: {reason}',
                'frequency': 0,
                'latency': 1000000,
                'invalid_config': True
            }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_project_dir = Path(temp_dir) / app_info['project_name']
            
            shutil.copytree(app_info['project_dir'], temp_project_dir)
            
            temp_config_file = temp_project_dir / app_info['config_file'].name
            with open(temp_config_file, 'r') as f:
                original_config = json.load(f)
            original_config.update(config)
            with open(temp_config_file, 'w') as f:
                json.dump(original_config, f, indent=4)
            
            synthesis_env = os.environ.copy()

            # Always override these so our choices aren't clobbered.
            synthesis_env['HLS_TARGET_MODE'] = 'hw_emu'

            # Ensure OPS_INSTALL_PATH is set so the Makefile include
            # $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app resolves
            # correctly even when the env was not sourced via the setup
            # scripts (e.g. when launched from probe.py or the system UI).
            if not synthesis_env.get('OPS_INSTALL_PATH'):
                ops_install = str(self.apps_base_dir.parent / 'OPS' / 'ops')
                synthesis_env['OPS_INSTALL_PATH'] = ops_install

            # Makefile presence is guaranteed by _find_available_apps.
            makefile_path = temp_project_dir / 'Makefile'
            makefile_content = makefile_path.read_text()
            if 'HLS_TARGET_MODE=' in makefile_content:
                makefile_content = makefile_content.replace('HLS_TARGET_MODE=hw', 'HLS_TARGET_MODE=hw_emu')
            else:
                makefile_content = 'HLS_TARGET_MODE=hw_emu\n' + makefile_content
            makefile_path.write_text(makefile_content)

            results = run_synthesis_with_progress(
                project_dir=temp_project_dir,
                config={**config, 'app_name': app_name},
                thread_id=threading.get_ident(),
                timeout=3600,
                env=synthesis_env,
            )

            if not results.get('synthesis_success', False):
                err = results.get('error_message', '')
                # Surface infrastructure failures in logs to make triage easier.
                if _is_infrastructure_error(err):
                    print(f"[HLSEnv] Infrastructure error (not a config failure): {err}")

            return results

    @staticmethod
    def _safe_int(val, default: int = 0) -> int:
        try:
            return int(float(str(val).strip()))
        except (TypeError, ValueError):
            return default

    def _parse_run_script_sizes(self, project_dir: Path) -> Optional[Tuple[int, int, int, int]]:
        """Parse representative (n_iter, nx, ny, nz) from run_script_hls.sh.

        Supports tuple forms used in this project:
          - "x,y,iters,batch"      (2D)
          - "x,y,z,iters,batch"    (3D)
        Selects the entry with largest n_iter, then largest grid volume.
        """
        run_script = project_dir / 'run_script_hls.sh'
        if not run_script.exists():
            return None

        try:
            text = run_script.read_text()
        except Exception:
            return None

        candidates: List[Tuple[int, int, int, int, int]] = []
        # Capture quoted numeric tuples inside parameter_sets.
        for tup in re.findall(r'"\s*([0-9]+(?:\s*,\s*[0-9]+){2,4})\s*"', text):
            parts = [self._safe_int(p) for p in tup.split(',')]
            if len(parts) == 4:
                nx, ny, n_iter, _batch = parts
                nz = 1
            elif len(parts) == 5:
                nx, ny, nz, n_iter, _batch = parts
            else:
                continue

            if nx <= 0 or ny <= 0 or nz <= 0 or n_iter <= 0:
                continue
            volume = nx * ny * nz
            candidates.append((n_iter, volume, nx, ny, nz))

        if not candidates:
            return None

        n_iter, _vol, nx, ny, nz = max(candidates, key=lambda t: (t[0], t[1]))
        return n_iter, nx, ny, nz

    def _infer_benchmark_sizes(self, app_info: Dict) -> Tuple[int, int, int, int]:
        """Infer benchmark sizes from run_script_hls.sh."""
        project_dir = app_info.get('project_dir')
        if not isinstance(project_dir, Path):
            project_dir = Path(project_dir) if project_dir else None

        app_name = app_info.get('app_name', '?')
        if not project_dir:
            raise RuntimeError(
                f"Cannot infer benchmark sizes for '{app_name}': missing project_dir in app_info."
            )

        run_sizes = self._parse_run_script_sizes(project_dir) if project_dir else None

        if run_sizes is not None:
            return run_sizes

        raise RuntimeError(
            f"Cannot infer benchmark sizes for '{app_name}'. "
            f"No parseable run_script_hls.sh parameter_sets were found under '{project_dir}'. "
            f"Add parameter_sets to run_script_hls.sh so (n_iter, nx, ny, nz) can be derived."
        )

    def _get_app_benchmark_sizes(self, app_info: Dict) -> Tuple[int, int, int, int]:
        """Return parsed (n_iter, nx, ny, nz) for the given app.

        Values are parsed from app-local artifacts and cached. Missing metadata
        is treated as a hard error to avoid silently training on wrong sizes.
        """
        app_name = app_info.get('app_name')
        if not app_name:
            raise ValueError(
                f"app_info has no 'app_name' key: {app_info}"
            )
        if app_name not in self._benchmark_size_cache:
            self._benchmark_size_cache[app_name] = self._infer_benchmark_sizes(app_info)
        return self._benchmark_size_cache[app_name]

    def _calculate_analytical_latency(self, config: Dict, frequency: float,
                                       app_info: Dict, ii: int = 1) -> float:
        """Estimate wall-clock runtime (µs) using the analytic latency model

          cycles = (n_iter / p) * (ceil(nx / V) * nz * (ny + D*p)) * II
        """
        if 'iter_par_factor' not in config:
            raise KeyError(f"config missing 'iter_par_factor': {config}")
        if 'vector_factor' not in config:
            raise KeyError(f"config missing 'vector_factor': {config}")
        p = config['iter_par_factor']
        V = config['vector_factor']

        stencil_features = app_info.get('stencil_features')
        if not stencil_features or 'stencils' not in stencil_features or not stencil_features['stencils']:
            raise ValueError(
                f"app_info for '{app_info.get('app_name')}' has no stencil features. "
                f"Ensure stencil JSON files exist in extracted_features/ before training."
            )
        # Use the largest stencil (most points) — stencils[0] is always the
        # identity/zero-offset S_00 placeholder and has max_radius=0.
        stencil = max(stencil_features['stencils'], key=lambda s: s.get('num_points', 0))
        if 'max_radius' not in stencil:
            raise KeyError(
                f"Stencil for '{app_info.get('app_name')}' missing 'max_radius': {stencil}"
            )
        D = stencil['max_radius']
        if D <= 0:
            raise ValueError(
                f"max_radius must be > 0 for '{app_info.get('app_name')}', "
                f"got {D} from stencil '{stencil.get('name')}' "
                f"(all stencils: {[(s.get('name'), s.get('max_radius'), s.get('num_points')) for s in stencil_features['stencils']]})"
            )

        n_iter, nx, ny, nz = self._get_app_benchmark_sizes(app_info)

        # Equation (8): Latency_ND = (n_iter/p) * (ceil(m1/V) * prod(m_i) * (m_N + D*p))
        latency_cycles = (n_iter / p) * (math.ceil(nx / V) * nz * (ny + D * p)) * ii
        runtime_us = latency_cycles / frequency  # freq in MHz → µs

        return runtime_us
    
    def _calculate_reward(self, metrics: Dict, config: Dict, app_info: Dict) -> np.ndarray:
        """Calculate multi-objective reward vector."""
        if not metrics.get('synthesis_success', False):
            if metrics.get('invalid_config', False):
                return np.array([-10.0] * 6, dtype=np.float32)
            return np.array([-1.0] * 6, dtype=np.float32)
        
        # Resource overflow: csynth completed but the design exceeds FPGA capacity.
        # Overflowed designs skip downstream timing metrics.
        # Compute per-resource rewards from the raw csynth counts so the network
        # receives a differentiated signal about *which* resources overflowed.
        if metrics.get('resource_overflow'):
            from fpga_specs import calculate_slr_utilization, U280_SPEC
            _util = calculate_slr_utilization(metrics, U280_SPEC)
            bram_reward = self._resource_reward(_util.get('bram', 0.0))
            dsp_reward  = self._resource_reward(_util.get('dsp',  0.0))
            lut_reward  = self._resource_reward(_util.get('lut',  0.0))
            ff_reward   = self._resource_reward(_util.get('ff',   0.0))
            _over_str = '  '.join(
                f"{r}={_util[r]*100:.1f}%"
                for r in ('bram', 'dsp', 'lut', 'ff')
                if _util.get(r, 0) > 1.0
            )
            print(
                f"[Reward] resource overflow for '{app_info.get('app_name','?')}': {_over_str}\n"
                f"  → rewards=[0.0000, {bram_reward:.4f}, {dsp_reward:.4f},"
                f" {lut_reward:.4f}, {ff_reward:.4f}, 0.5000]  (runtime=0, freq=neutral)"
            )
            # runtime=0.0 (can't place), freq=0.5 (neutral/unknown)
            return np.array([0.0, bram_reward, dsp_reward, lut_reward, ff_reward, 0.5],
                            dtype=np.float32)

        if 'frequency' not in metrics:
            raise RuntimeError("Synthesis succeeded but 'frequency' metric is missing")
        frequency = metrics['frequency']
        if 'initiation_interval' not in metrics or not metrics['initiation_interval']:
            raise RuntimeError(
                f"Synthesis succeeded but 'initiation_interval' is missing or zero in metrics: {metrics}"
            )
        ii = metrics['initiation_interval']

        # Extract model parameters for logging
        p = config['iter_par_factor']
        V = config['vector_factor']
        _all_stencils = app_info['stencil_features']['stencils']
        _main_stencil = max(_all_stencils, key=lambda s: s.get('num_points', 0))
        D = _main_stencil['max_radius']
        _stencil_summary = "  ".join(
            f"{s.get('name','?')}(pts={s.get('num_points','?')},r={s.get('max_radius','?')})"
            for s in _all_stencils
        )
        print(
            f"[Reward] stencil selection for '{app_info.get('app_name','?')}':\n"
            f"  candidates: {_stencil_summary}\n"
            f"  → main stencil (most points): '{_main_stencil.get('name','?')}'  "
            f"num_points={_main_stencil.get('num_points','?')}  max_radius(D)={D}"
        )
        n_iter, nx, ny, nz = self._get_app_benchmark_sizes(app_info)

        # Runtime estimate uses the analytical formula with csynth II.
        analytical_runtime_us = self._calculate_analytical_latency(config, frequency, app_info, ii=ii)

        # Normalise against a notional serial baseline (V=1, p=1, II=1) so that
        # even the minimum valid config (V=2, p=1) receives a non-zero reward (~0.5),
        # and the gradient spans the full reachable parallelism space.
        # V=1 is not a valid synthesisable config but is used here purely as a
        # normalisation anchor — it never runs through HLS.
        worst_case_us = self._calculate_analytical_latency(
            {'iter_par_factor': 1, 'vector_factor': 1}, frequency, app_info, ii=1
        )
        runtime_reward = max(0.0, min(1.0, 1.0 - analytical_runtime_us / worst_case_us))

        latency_cycles = (n_iter / p) * (math.ceil(nx / V) * nz * (ny + D * p)) * ii
        
        from fpga_specs import calculate_resource_utilization, U280_SPEC
        resource_util = calculate_resource_utilization(metrics, U280_SPEC)
        bram_util = resource_util['bram']
        dsp_util = resource_util['dsp']
        ff_util = resource_util['ff']
        lut_util = resource_util['lut']
        
        # Soft-cliff resource reward: flat 1.0 in the safe zone, linear warning
        # ramp as utilisation approaches 100%, hard 0 once over capacity.
        #
        #   util in [0, SAFE]        → 1.0          (no penalty, comfortable)
        #   util in (SAFE, 1.0]      → linear 1→0   (gradient signal near limit)
        #   util > 1.0               → 0.0           (overflow: won't fit)
        #
        # This avoids both failure modes:
        #  - old '1-util': penalised 40% as bad as 90%, steering away from
        #    high-parallelism configs that are valid and fast.
        #  - pure binary: no gradient signal until overflow; network can't
        #    learn that 95% DSP is riskier than 40% DSP.
        bram_reward = self._resource_reward(bram_util)
        dsp_reward  = self._resource_reward(dsp_util)
        lut_reward  = self._resource_reward(lut_util)
        ff_reward   = self._resource_reward(ff_util)
        
        # Frequency reward from csynth timing slack.
        #
        # Formula (same for both sources):
        #   norm_slack = slack_ns / target_period_ns
        #   freq_reward = clip(0.5 + norm_slack * 2,  0.0, 1.0)
        #
        #   slack > 0  → timing met, reward > 0.5  (max 1.0 at +target/4 margin)
        #   slack = 0  → exactly on target          → 0.5
        #   slack < 0  → timing violated             → < 0.5  (0.0 at −target/4)
        target_ns  = 1000.0 / 400.0  # 2.5 ns  (400 MHz target)
        csynth_slack = metrics.get('timing_slack_ns')
        if csynth_slack is not None:
            slack_ns  = csynth_slack
            freq_src  = f"csynth slack={csynth_slack:+.3f}ns"
        else:
            slack_ns  = None
            freq_src  = "unknown (neutral 0.5)"
        if slack_ns is not None:
            norm_slack  = slack_ns / target_ns
            freq_reward = max(0.0, min(1.0, 0.5 + norm_slack * 2))
        else:
            freq_reward = 0.5
        _rt_detail = f"formula={latency_cycles:.0f} cyc"
        print(
            f"[Synth] {app_info.get('app_name','?')}  p={p}  V={V}  MVF={config.get('mem_vector_factor','?')}"
            f"  D={D}  II={ii}  n_iter={n_iter}  nx={nx}  ny={ny}  nz={nz}\n"
            f"  depth={config.get('maxi_depth','?')}"
            f"  rb={config.get('maxi_read_burst_length','?')}  wb={config.get('maxi_write_burst_length','?')}"
            f"  ro={config.get('num_read_outstanding','?')}  wo={config.get('num_write_outstanding','?')}"
            f"  axis={config.get('axis_interconnect_buff_size','?')}  hls={config.get('hls_interconnect_buff_size','?')}\n"
            f"  runtime (analytical): {_rt_detail}  freq={frequency:.1f}MHz"
            f"  → runtime={analytical_runtime_us:.2f}µs  reward={runtime_reward:.4f}\n"
            f"  freq    ({freq_src}): reward={freq_reward:.4f}\n"
            f"  BRAM (csynth): util={bram_util*100:.1f}%  {self._util_tag(bram_util)}  reward={bram_reward:.4f}\n"
            f"  DSP  (csynth): util={dsp_util*100:.1f}%  {self._util_tag(dsp_util)}  reward={dsp_reward:.4f}\n"
            f"  LUT  (csynth): util={lut_util*100:.1f}%  {self._util_tag(lut_util)}  reward={lut_reward:.4f}\n"
            f"  FF   (csynth): util={ff_util*100:.1f}%  {self._util_tag(ff_util)}  reward={ff_reward:.4f}\n"
            f"  → rewards=[{runtime_reward:.4f}, {bram_reward:.4f}, {dsp_reward:.4f},"
            f" {lut_reward:.4f}, {ff_reward:.4f}, {freq_reward:.4f}]"
        )

        return np.array([runtime_reward, bram_reward, dsp_reward, lut_reward, ff_reward, freq_reward], dtype=np.float32)
    
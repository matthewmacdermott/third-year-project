"""Launch multi-node DQN training: one trainer job + N worker jobs via SLURM.

Architecture
------------
  Trainer job  (gpu partition, long-running):
    - Reads experiences from shared ExperienceStore (SQLite on shared fs)
    - Performs gradient updates continuously
    - Saves shared checkpoint (latest.pth) that workers reload

  Worker jobs  (fpgasynthesis partition, shorter, SLURM array):
    - Load latest checkpoint at start
    - Run Vitis HLS synthesis batches
    - Write (state, action, reward) to ExperienceStore
    - Reload checkpoint periodically

Shared state (all in the run_dir):
  experience_store.db   — SQLite, concurrent-safe (WAL mode)
  latest.pth            — checkpoint written by trainer, read by workers
  models/dqn_best_*.pth — best checkpoints kept by trainer
"""

import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .fpga_builder import JobState, JobResult

# ── SBATCH templates ──────────────────────────────────────────────────────────

_TRAINER_TEMPLATE = """\
#!/bin/sh
#SBATCH -J mn_trainer_{tag}
#SBATCH -o {log_path}
#SBATCH -t {wall_time}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -A hpc-prf-acgasm
#SBATCH -p normal
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user matthew.macdermott@warwick.ac.uk

set -e
# Trainer only needs Python + PyTorch — no Vitis/OPS required
module reset
module load lang/Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${{PYTHONPATH:-}}"
python -c "import torch; print('torch', torch.__version__)" || pip install torch numpy

cd "{trainer_dir}"
python -u trainer_node.py \\
  --store          '{store_path}' \\
  --checkpoint     '{checkpoint_path}' \\
  --run-dir        '{run_dir}' \\
  --max-time       {max_time_h} \\
  --save-freq      {save_freq} \\
  --poll-interval  {poll_interval}{pretrain_line}{load_model_line}{reset_epsilon_line}
echo "=== Trainer complete ==="
"""

_WORKER_TEMPLATE = """\
#!/bin/sh
#SBATCH -J mn_worker_{tag}_%a
#SBATCH -o {log_path}_%a.out
#SBATCH -t {wall_time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH -A hpc-prf-acgasm
#SBATCH -q fpgasynthesis
#SBATCH -p normal
#SBATCH --array=0-{last_worker_idx}
#SBATCH --mail-type FAIL
#SBATCH --mail-user matthew.macdermott@warwick.ac.uk

set -e
unset OPS_HLS_ARTIFACT_DIR
source "{source_script}"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${{PYTHONPATH:-}}"
python -c "import torch; print('torch', torch.__version__)" || pip install torch numpy

cd "{trainer_dir}"
python -u worker.py \\
  --apps-dir    '{apps_dir}' \\
  --store       '{store_path}' \\
  --checkpoint  '{checkpoint_path}' \\
  --batch-size  {batch_size} \\
  --max-time    {worker_time_h} \\
  --worker-id   $SLURM_ARRAY_TASK_ID
echo "=== Worker $SLURM_ARRAY_TASK_ID complete ==="
"""

# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class MultiNodeConfig:
    # Time budgets
    trainer_time_h: float = 12.0   # Wall time for the trainer SLURM job
    worker_time_h:  float = 6.0    # Wall time for each worker SLURM job

    # Parallelism
    num_workers:        int = 4    # Number of worker SLURM array jobs
    worker_batch_size:  int = 4    # Parallel synthesis jobs per worker node

    # Trainer options
    save_freq:        int   = 20
    poll_interval_s:  float = 30.0
    load_model:       Optional[str] = None
    reset_epsilon:    Optional[float] = None
    pretrain_stores:  Optional[List[str]] = None
    pretrain_epochs:  int = 3

    def validate(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.worker_batch_size < 1:
            raise ValueError("worker_batch_size must be >= 1")


# ── Launcher ──────────────────────────────────────────────────────────────────

class MultiNodeLauncher:
    """Launch trainer + worker SLURM jobs for multi-node DQN training."""

    def __init__(self, project_root: Path, source_script: Optional[str] = None):
        self.project_root = project_root
        self.trainer_dir  = project_root / "project_files" / "trainer"
        self.apps_dir     = project_root / "codegen_apps"
        self.jobs_dir     = project_root / "project_files" / "system" / "_jobs"
        self._train_dir   = self.jobs_dir  # resolved per-run in launch()
        scripts_dir       = project_root / "scripts"

        if source_script:
            self.source_script = source_script
        else:
            self.source_script = str(scripts_dir / "source_noctuna2_vitis_2023_2_ops.sh")
            for name in (
                "source_noctuna2_vitis_2023_2_ops.sh",
                "source_noctuna2_vitis_2022_2_ops.sh",
                "source_owl_vitis_2022_2_ops.sh",
            ):
                if (scripts_dir / name).exists():
                    self.source_script = str(scripts_dir / name)
                    break

    def launch(self, config: MultiNodeConfig, tag: str = "mn") -> dict:
        """Submit trainer and worker jobs.  Returns dict with job IDs and paths."""
        config.validate()
        if not shutil.which("sbatch"):
            raise EnvironmentError("sbatch not found – SLURM unavailable on this node.")

        ts      = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.project_root / "project_files" / "runs" / f"mn_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # Each run gets its own subdirectory named after the tag
        run_jobs_dir = self.jobs_dir / f"{tag}_{ts}"
        run_jobs_dir.mkdir(parents=True, exist_ok=True)

        store_path      = str(run_dir / "experience_store.db")
        checkpoint_path = str(run_dir / "latest.pth")

        # ── Trainer job ───────────────────────────────────────────────────
        trainer_log  = run_jobs_dir / f"trainer_{tag}_{ts}.out"
        trainer_wt   = f"{int(config.trainer_time_h)}:{int((config.trainer_time_h % 1)*60):02d}:00"
        load_line          = f" \\\n  --load-model '{config.load_model}'" if config.load_model else ""
        reset_epsilon_line = f" \\\n  --reset-epsilon {config.reset_epsilon}" if config.reset_epsilon is not None else ""
        pretrain_line      = (
            " \\\n  --pretrain-store " + " ".join(f"'{s}'" for s in config.pretrain_stores)
            + f" \\\n  --pretrain-epochs {config.pretrain_epochs}"
        ) if config.pretrain_stores else ""

        trainer_script = textwrap.dedent(_TRAINER_TEMPLATE.format(
            tag=tag, log_path=trainer_log,
            wall_time=trainer_wt,
            source_script=self.source_script,
            trainer_dir=self.trainer_dir,
            store_path=store_path,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            max_time_h=config.trainer_time_h,
            save_freq=config.save_freq,
            poll_interval=config.poll_interval_s,
            pretrain_line=pretrain_line,
            load_model_line=load_line,
            reset_epsilon_line=reset_epsilon_line,
        ))
        trainer_sh = run_jobs_dir / f"trainer_{tag}_{ts}.sh"
        trainer_sh.write_text(trainer_script)
        trainer_sh.chmod(0o755)

        r = subprocess.run(["sbatch", str(trainer_sh)],
                           capture_output=True, text=True, timeout=30, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"sbatch (trainer) failed: {r.stderr.strip()}")
        trainer_job_id = r.stdout.strip().split()[-1]

        # ── Worker array job ──────────────────────────────────────────────
        # Each worker uses worker_batch_size * 4 CPUs (vitis_hls threads)
        cpus_per_worker = min(128, config.worker_batch_size * 4 + 4)
        mem_per_worker  = f"{max(32, config.worker_batch_size * 3 + 8)}G"
        worker_wt       = f"{int(config.worker_time_h)}:{int((config.worker_time_h % 1)*60):02d}:00"
        worker_log_pfx  = str(run_jobs_dir / f"worker_{tag}_{ts}")

        worker_script = textwrap.dedent(_WORKER_TEMPLATE.format(
            tag=tag,
            log_path=worker_log_pfx,
            wall_time=worker_wt,
            cpus=cpus_per_worker,
            mem=mem_per_worker,
            last_worker_idx=config.num_workers - 1,
            source_script=self.source_script,
            trainer_dir=self.trainer_dir,
            apps_dir=self.apps_dir,
            store_path=store_path,
            checkpoint_path=checkpoint_path,
            batch_size=config.worker_batch_size,
            worker_time_h=config.worker_time_h,
        ))
        worker_sh = run_jobs_dir / f"worker_{tag}_{ts}.sh"
        worker_sh.write_text(worker_script)
        worker_sh.chmod(0o755)

        r = subprocess.run(["sbatch", str(worker_sh)],
                           capture_output=True, text=True, timeout=30, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"sbatch (worker array) failed: {r.stderr.strip()}")
        worker_job_id = r.stdout.strip().split()[-1]

        print(
            f"\n[MultiNode] Submitted successfully\n"
            f"  Trainer job:  {trainer_job_id}  → {trainer_log}\n"
            f"  Worker array: {worker_job_id} ({config.num_workers} workers)"
            f"  → {worker_log_pfx}_*.out\n"
            f"  Run dir:      {run_dir}\n"
            f"  Store:        {store_path}\n"
            f"  Checkpoint:   {checkpoint_path}"
        )

        return {
            "trainer_job_id": trainer_job_id,
            "worker_job_id":  worker_job_id,
            "run_dir":        str(run_dir),
            "store_path":     store_path,
            "checkpoint_path": checkpoint_path,
            "trainer_log":    str(trainer_log),
        }

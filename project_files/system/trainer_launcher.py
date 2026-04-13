"""Launch DQN training runs via SLURM."""

import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .fpga_builder import JobState, JobResult

_TRAIN_SBATCH_TEMPLATE = """\
#!/bin/sh
#SBATCH -J dqn_train_{tag}
#SBATCH -o {log_path}
#SBATCH -t {wall_time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH -A hpc-prf-acgasm
#SBATCH -p normal
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user matthew.macdermott@warwick.ac.uk

set -e
unset OPS_HLS_ARTIFACT_DIR
source "{source_script}"

# Make user-installed packages (torch etc.) visible inside the venv
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${{PYTHONPATH:-}}"
echo "PYTHONPATH set"

python -c "import torch; print('torch', torch.__version__)" || {{ echo "torch not found, installing..."; pip install torch numpy; }}
cd "{trainer_dir}"

python -u main.py \\
{args_block}
echo "=== Training complete ==="
"""


@dataclass
class TrainConfig:
    num_runs: Optional[int] = None
    max_time_h: Optional[float] = None
    batch_size: int = 4
    save_freq: int = 10
    load_model: Optional[str] = None

    def validate(self) -> None:
        if self.num_runs is None and self.max_time_h is None:
            raise ValueError("Must specify either num_runs or max_time_h.")

    def to_args(self, run_dir: str) -> str:
        parts: List[str] = []
        if self.num_runs is not None:
            parts.append(f"  --num-runs {self.num_runs} \\")
        if self.max_time_h is not None:
            parts.append(f"  --max-time {self.max_time_h} \\")
        parts.append(f"  --batch-size {self.batch_size} \\")
        parts.append(f"  --save-freq {self.save_freq} \\")
        if self.load_model:
            parts.append(f"  --load-model '{self.load_model}' \\")
        parts.append(f"  --run-dir '{run_dir}' \\")
        if parts:
            parts[-1] = parts[-1].rstrip(" \\")
        return "\n".join(parts)


class TrainerLauncher:
    def __init__(self, project_root: Path, source_script: Optional[str] = None):
        self.project_root = project_root
        self.trainer_dir = project_root / "project_files" / "trainer"
        self.jobs_dir = project_root / "project_files" / "system" / "_jobs"
        scripts_dir = project_root / "scripts"
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

    def launch(self, config: TrainConfig, tag: str = "run") -> JobResult:
        config.validate()
        if not shutil.which("sbatch"):
            raise EnvironmentError("sbatch not found \u2013 SLURM unavailable on this node.")

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.jobs_dir / f"train_{tag}_{ts}.out"

        wt_h = (config.max_time_h + 0.5) if config.max_time_h else min(72, (config.num_runs or 100) * 0.5 + 1)
        wall_time = f"{int(wt_h)}:{int((wt_h % 1) * 60):02d}:00"

        run_dir = self.project_root / "project_files" / "runs" / ts
        # Scale SLURM resources with batch_size.
        # vitis_hls uses ~4 threads per synthesis; +4 for the Python trainer process.
        # Memory: ~3 GB per vitis_hls instance + 8 GB Python/model overhead.
        n_jobs = config.batch_size
        slurm_cpus = min(128, n_jobs * 4 + 4)
        slurm_mem  = f"{max(64, n_jobs * 3 + 8)}G"
        script_text = textwrap.dedent(_TRAIN_SBATCH_TEMPLATE.format(
            tag=tag, log_path=log_path, wall_time=wall_time,
            cpus=slurm_cpus, mem=slurm_mem,
            source_script=self.source_script,
            trainer_dir=self.trainer_dir,
            args_block=config.to_args(str(run_dir)),
        ))
        sh = self.jobs_dir / f"train_{tag}_{ts}.sh"
        sh.write_text(script_text)
        sh.chmod(0o755)

        r = subprocess.run(["sbatch", str(sh)], capture_output=True, text=True, timeout=30, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"sbatch failed: {r.stderr.strip()}")

        job_id = r.stdout.strip().split()[-1]
        return JobResult(job_id=job_id, state=JobState.PENDING, log_path=log_path)


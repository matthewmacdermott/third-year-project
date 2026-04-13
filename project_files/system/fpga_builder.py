"""Submit and monitor FPGA HW build/run jobs via SLURM."""

import subprocess
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from .stencil_selector import StencilProject


class JobState(Enum):
    PENDING   = auto()
    RUNNING   = auto()
    COMPLETED = auto()
    FAILED    = auto()
    UNKNOWN   = auto()

    @classmethod
    def from_sacct(cls, raw: str) -> "JobState":
        r = raw.strip().upper()
        if r in ("PENDING", "CONFIGURING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUE_HOLD"):
            return cls.PENDING
        if r in ("RUNNING", "COMPLETING"):
            return cls.RUNNING
        if r == "COMPLETED":
            return cls.COMPLETED
        if r in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
            return cls.FAILED
        return cls.UNKNOWN


@dataclass
class JobResult:
    job_id: str
    state: JobState
    log_path: Path


class FPGABuilder:
    def __init__(self, project_root: Path, source_script: Optional[str] = None):
        self.project_root = project_root
        self.source_script = source_script or self._detect_source_script()
        self.jobs_dir  = project_root / "project_files" / "system" / "_jobs"
        self._build_dir = self.jobs_dir  # resolved per-job in _submit()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _detect_source_script(self) -> str:
        scripts = self.project_root / "scripts"
        for name in (
            "source_noctuna2_vitis_2023_2_ops.sh",
            "source_noctuna2_vitis_2022_2_ops.sh",
            "source_owl_vitis_2022_2_ops.sh",
        ):
            p = scripts / name
            if p.exists():
                return str(p)
        return str(scripts / "source_noctuna2_vitis_2023_2_ops.sh")

    def _log_path(self, kind: str, stencil: StencilProject) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        job_dir = self.jobs_dir / f"{kind}_{stencil.app_name}_{stencil.board}_{ts}"
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir / f"{kind}_{stencil.app_name}_{stencil.board}_{ts}.out"

    def _submit(self, kind: str, stencil: StencilProject) -> "JobResult":
        is_build = kind == "build"
        log = self._log_path(kind, stencil)
        resources = (
            "#SBATCH -t 12:00:00\n"
            "#SBATCH --cpus-per-task=8\n"
            "#SBATCH --mem=64G\n"
            "#SBATCH -q fpgasynthesis\n"
            "#SBATCH -p normal\n"
        ) if is_build else (
            "#SBATCH -t 2:00:00\n"
            "#SBATCH --partition=fpga\n"
            "#SBATCH --constraint=xilinx_u280_xrt2.16\n"
            "#SBATCH --exclusive\n"
        )
        script = (
            "#!/bin/sh\n"
            f"#SBATCH -J {kind}_{stencil.app_name}_{stencil.board}\n"
            f"#SBATCH -o {log}\n"
            "#SBATCH -A hpc-prf-acgasm\n"
            "#SBATCH --mail-type FAIL,END\n"
            "#SBATCH --mail-user matthew.macdermott@warwick.ac.uk\n"
            + resources +
            "set -e\n"
            "unset OPS_HLS_ARTIFACT_DIR\n"
            f'source "{self.source_script}"\n'
            f'cd "{stencil.project_dir}"\n'
            + ("make clean\nmake\n" if is_build else "make run_hls_app\n") +
            f"echo '=== {kind} complete ==='\n"
        )
        self._build_dir.mkdir(parents=True, exist_ok=True)
        sh = log.parent / f"{kind}_{stencil.app_name}_{stencil.board}.sh"
        sh.write_text(script)
        sh.chmod(0o755)
        try:
            r = subprocess.run(["sbatch", str(sh)], capture_output=True, text=True,
                               timeout=30, check=True)
        except FileNotFoundError:
            raise EnvironmentError("sbatch not found - SLURM unavailable on this node.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"sbatch failed: {e.stderr.strip()}")
        job_id = r.stdout.strip().split()[-1]
        return JobResult(job_id=job_id, state=JobState.PENDING, log_path=log)

    def submit_build(self, stencil: StencilProject) -> "JobResult":
        return self._submit("build", stencil)

    def submit_run(self, stencil: StencilProject) -> "JobResult":
        return self._submit("run", stencil)

    def poll_state(self, job_id: str) -> JobState:
        try:
            r = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                capture_output=True, text=True, timeout=15,
            )
            lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
            if lines:
                return JobState.from_sacct(lines[0].split("|")[0])
        except Exception:
            pass
        return JobState.UNKNOWN

    def wait_for_job(self, result: "JobResult", poll_s: int = 120,
                     timeout_h: float = 6.0) -> "JobResult":
        deadline = time.time() + timeout_h * 3600
        start = time.time()
        while time.time() < deadline:
            state = self.poll_state(result.job_id)
            print("    [{:>3}m] Job {}: {}".format(
                int(time.time() - start) // 60, result.job_id, state.name))
            if state in (JobState.COMPLETED, JobState.FAILED):
                return JobResult(result.job_id, state, result.log_path)
            time.sleep(poll_s)
        return JobResult(result.job_id, JobState.UNKNOWN, result.log_path)

    @staticmethod
    def squeue_summary(job_id: Optional[str] = None) -> str:
        cmd = ["squeue", "--me", "--format=%.10i %.20j %.8T %.12M %.8l %.6D %R"]
        if job_id:
            cmd += ["--job", job_id]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
        except Exception:
            return ""

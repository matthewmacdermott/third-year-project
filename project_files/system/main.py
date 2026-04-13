#!/usr/bin/env python3
"""HLS Stencil Optimiser - interactive orchestration system."""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_SYSTEM_DIR        = Path(__file__).resolve().parent
_PROJECT_FILES_DIR = _SYSTEM_DIR.parent
_PROJECT_ROOT      = _PROJECT_FILES_DIR.parent

from . import ui
from .fpga_builder       import FPGABuilder, JobState
from .model_manager      import ModelInfo, discover_models
from .recommender_runner import RecommenderRunner, apply_config_to_project, format_recommendations
from .stencil_selector   import StencilProject, discover_stencils
from .trainer_launcher      import TrainConfig, TrainerLauncher
from .multi_node_launcher   import MultiNodeConfig, MultiNodeLauncher


class Orchestrator:
    def __init__(self, project_root=_PROJECT_ROOT):
        self.project_root = project_root
        self.codegen_apps = project_root / "codegen_apps"
        self.builder     = FPGABuilder(project_root)
        self.trainer     = TrainerLauncher(project_root)
        self.mn_launcher = MultiNodeLauncher(project_root)
        self.recommender = RecommenderRunner(project_root)
        self._selected_model   = None  # type: Optional[ModelInfo]
        self._selected_stencil = None  # type: Optional[StencilProject]

    def _show_status(self):
        m = self._selected_model.path.name if self._selected_model else ui.dim("(none)")
        s = ("{}/{}".format(self._selected_stencil.app_name, self._selected_stencil.board)
             if self._selected_stencil else ui.dim("(none)"))
        print("  Model  : " + ui.cyan(m))
        print("  Stencil: " + ui.cyan(s))

    def run(self):
        ui.banner("HLS Stencil Optimiser")
        print("  Project root: " + ui.dim(str(self.project_root)))
        print()
        while True:
            self._selected_model   = None
            self._selected_stencil = None
            choice = ui.menu(
                "Main Menu",
                [
                    ("Run recommendation -> build -> results", "run_flow"),
                    ("Train a new model",                      "train"),
                    ("Evaluate model across all stencils",     "evaluate_all"),
                    ("Validate all: submit HW builds from eval results", "validate_all"),
                    ("Run HW performance profile on a stencil", "hw_run"),
                    ("Probe: run synthesis on a stencil + config", "probe"),
                ],
                allow_back=False,
                zero_exit=True,
                exit_label="Exit",
            )
            if choice is None:
                print("\nGoodbye.\n")
                break
            {
                "run_flow":     self._flow_run,
                "train":        self._flow_train,
                "evaluate_all": self._flow_evaluate_all,
                "validate_all": self._flow_validate_all,
                "hw_run":       self._flow_hw_run,
                "probe":        self._flow_probe,
            }[choice]()

    def _flow_select_model(self):
        ui.section("Select Trained Model")
        models = discover_models()
        if not models:
            from .model_manager import MODELS_DIR
            ui.warn("No .pth files found in " + str(MODELS_DIR))
            ui.info("Drop .pth files there and re-run.")
            return
        chosen = ui.menu("Choose a model", [(m.label, m) for m in models])
        if chosen:
            self._selected_model = chosen
            ui.success("Selected: " + str(chosen.path))

    def _flow_select_stencil(self):
        ui.section("Select Target Stencil")
        stencils = discover_stencils(self.codegen_apps)
        if not stencils:
            ui.warn("No stencil projects found under " + str(self.codegen_apps))
            return
        chosen = ui.menu("Choose a stencil", [(s.label, s) for s in stencils])
        if chosen:
            self._selected_stencil = chosen
            ui.success("Selected: {}/{} ({})".format(
                chosen.app_name, chosen.board, chosen.project_dir))

    def _flow_run(self):
        ui.banner("Run Recommendation -> Build -> Results")

        if self._selected_model is None:
            ui.warn("No model selected.")
            self._flow_select_model()
            if self._selected_model is None:
                return

        if self._selected_stencil is None:
            ui.warn("No stencil selected.")
            self._flow_select_stencil()
            if self._selected_stencil is None:
                return

        model, stencil = self._selected_model, self._selected_stencil
        ui.info("Model  : " + str(model.path))
        ui.info("Stencil: {}/{}".format(stencil.app_name, stencil.board))

        strategy = ui.menu(
            "Recommendation strategy",
            [
                ("Balanced  - equal weight on resources & performance", "balanced"),
                ("Resource  - minimise BRAM / DSP / LUT / FF",          "resource"),
                ("Performance - maximise frequency, minimise latency",   "performance"),
            ],
        )
        if strategy is None:
            return

        try:
            top_k = int(ui.prompt("How many recommendations to show?", default="5"))
        except ValueError:
            top_k = 5

        ui.section("Running Recommender")
        export_path = stencil.project_dir / "recommended_config.json"
        try:
            recs = self.recommender.run_to_json(
                stencil, model.path,
                output_json=export_path,
                top_k=top_k,
                strategy=strategy,
            )
        except Exception as e:
            ui.error("Recommender failed: " + str(e))
            return

        print(format_recommendations(recs))
        print()
        ui.success("Best config written to: " + str(export_path))

        best_config = recs[0]["config"]
        ui.section("Best Recommended Configuration")
        for k, v in best_config.items():
            print("    {:<45} = {}".format(k, v))
        print()

        next_action = ui.menu(
            "What would you like to do with this configuration?",
            [
                ("Probe  - run csynth + reward inspection (a few minutes)", "probe"),
                ("Build  - apply config and submit full HW build job",      "build"),
                ("Skip   - do nothing",                                     "skip"),
            ],
        )
        if next_action is None or next_action == "skip":
            return

        if next_action == "probe":
            probe_py = _PROJECT_FILES_DIR / "trainer" / "probe.py"
            cmd = [sys.executable, str(probe_py), stencil.app_name]
            for param, flag in [
                ("vector_factor",              "--vf"),
                ("mem_vector_factor",           "--mvf"),
                ("iter_par_factor",             "--ipf"),
                ("maxi_depth",                  "--depth"),
                ("maxi_read_burst_length",       "--rb"),
                ("maxi_write_burst_length",      "--wb"),
                ("num_read_outstanding",         "--ro"),
                ("num_write_outstanding",        "--wo"),
                ("axis_interconnect_buff_size",  "--axis"),
                ("hls_interconnect_buff_size",   "--hls"),
            ]:
                if param in best_config:
                    cmd += [flag, str(best_config[param])]
            ui.section("Running probe on recommended config")
            ui.info("Command: " + " ".join(cmd))
            print()
            try:
                subprocess.run(cmd, check=False)
            except FileNotFoundError:
                ui.error("probe.py not found at: " + str(probe_py))
            return

        # next_action == "build"
        try:
            apply_config_to_project(best_config, stencil)
            ui.success("Config applied to " + str(stencil.config_file))
        except Exception as e:
            ui.error("Failed to apply config: " + str(e))
            return

        try:
            build_job = self.builder.submit_build(stencil)
            ui.success("Build job submitted: " + build_job.job_id)
            ui.info("Log: " + str(build_job.log_path))
            print(self.builder.squeue_summary(build_job.job_id))
        except Exception as e:
            ui.error("Failed to submit build: " + str(e))
            return

        if not ui.confirm("Wait for build, then auto-submit run job?"):
            return

        ui.info("Polling SLURM (Ctrl-C to stop)...")
        try:
            build_job = self.builder.wait_for_job(build_job)
        except KeyboardInterrupt:
            ui.warn("Stopped waiting. Job still running in background.")
            return

        if build_job.state != JobState.COMPLETED:
            ui.error("Build did not complete. Check log: " + str(build_job.log_path))
            return
        ui.success("Build complete.")

        try:
            run_job = self.builder.submit_run(stencil)
            ui.success("Run job submitted: " + run_job.job_id)
            ui.info("Log: " + str(run_job.log_path))
            print(self.builder.squeue_summary(run_job.job_id))
        except Exception as e:
            ui.error("Failed to submit run: " + str(e))

    def _flow_hw_run(self):
        ui.banner("HW Performance Profile")
        ui.info("Runs the compiled bitstream on the FPGA and saves timing CSVs.")
        ui.info("Assumes HW synthesis has already been completed.")
        print()

        stencils = discover_stencils(self.codegen_apps)
        if not stencils:
            ui.warn("No stencil projects found under " + str(self.codegen_apps))
            return
        stencil = ui.menu("Choose a stencil", [(s.label, s) for s in stencils])
        if stencil is None:
            return
        ui.success("Selected: {}/{}".format(stencil.app_name, stencil.board))

        # Check bitstream exists (accept any .xclbin in the hw build dir)
        hw_dir = stencil.project_dir / "hls" / "build" / "hw"
        xcl_matches = list(hw_dir.glob("*.xclbin")) if hw_dir.exists() else []
        if not xcl_matches:
            ui.error("No .xclbin found in: " + str(hw_dir))
            ui.error("Run HW synthesis first (from the main flow or manually via make).")
            return
        xclbin = xcl_matches[0]
        ui.success("Found bitstream: " + str(xclbin))

        profile_dir = stencil.project_dir / "hls" / "profile_data" / "hw"

        # Build the SBATCH script — mirrors the existing sbatch_run_script_2023_2.sh
        # but uses a dynamic log path and timestamp-tagged job name.
        ts  = time.strftime("%Y%m%d_%H%M%S")
        hw_jobs_dir = self.builder.jobs_dir / f"hwrun_{stencil.app_name}_{ts}"
        hw_jobs_dir.mkdir(parents=True, exist_ok=True)
        log = hw_jobs_dir / f"hwrun_{stencil.app_name}_{ts}.out"
        script = (
            "#!/bin/sh\n"
            f"#SBATCH -J hwrun_{stencil.app_name}_{ts}\n"
            f"#SBATCH -o {log}\n"
            "#SBATCH -A hpc-prf-acgasm\n"
            "#SBATCH --mail-type FAIL,END\n"
            "#SBATCH --mail-user matthew.macdermott@warwick.ac.uk\n"
            "#SBATCH -t 4:00:00\n"
            "#SBATCH --partition=fpga\n"
            "#SBATCH --constraint=xilinx_u280_xrt2.16\n"
            "#SBATCH --exclusive\n"
            "set -e\n"
            f'source "{self.builder.source_script}"\n'
            f'cd "{stencil.project_dir}"\n'
            "make run_hls_app\n"
            f"echo '=== HW profiling complete ==='\n"
        )
        sh = hw_jobs_dir / f"hwrun_{stencil.app_name}_{ts}.sh"
        sh.write_text(script)
        sh.chmod(0o755)

        ui.section("Summary")
        ui.info(f"Stencil  : {stencil.app_name}/{stencil.board}")
        ui.info(f"Grid sizes: defined in run_script_hls.sh for platform u280")
        ui.info(f"Results  → {profile_dir}/")
        ui.info(f"Log      → {log}")
        print()

        if not ui.confirm("Submit HW run job?", default=True):
            return

        try:
            r = subprocess.run(["sbatch", str(sh)], capture_output=True, text=True,
                               timeout=30, check=False)
            if r.returncode != 0:
                ui.error("sbatch failed: " + r.stderr.strip())
                return
            job_id = r.stdout.strip().split()[-1]
            ui.success(f"HW run job submitted: {job_id}")
            ui.info("CSVs will appear in: " + str(profile_dir))
            print(self.builder.squeue_summary(job_id))
        except Exception as e:
            ui.error("Failed to submit: " + str(e))
            return

        if not ui.confirm("Watch for CSV results as the job runs?", default=True):
            return

        ui.info("Polling SLURM every 30 s and printing CSVs as they appear (Ctrl-C to stop)...")
        seen_csvs: set = set()

        def _print_csv(csv_path):
            print(f"\n  {csv_path.name}")
            print("  " + "-" * (len(csv_path.name) + 2))
            try:
                lines = csv_path.read_text().splitlines()
                for line in lines[:5]:
                    print("  " + line)
                if len(lines) > 5:
                    print(f"  ... ({len(lines) - 5} more rows)")
            except Exception as exc:
                ui.warn(f"Could not read {csv_path.name}: {exc}")

        try:
            while True:
                state = self.builder.poll_state(job_id)
                # Print any new CSVs that have appeared
                if profile_dir.exists():
                    for csv in sorted(profile_dir.glob("*.csv")):
                        if csv not in seen_csvs:
                            seen_csvs.add(csv)
                            _print_csv(csv)
                if state in (JobState.COMPLETED, JobState.FAILED):
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            ui.warn("Stopped watching. Job still running in background.")
            return

        # Final pass — catch any CSVs written right at completion
        if profile_dir.exists():
            for csv in sorted(profile_dir.glob("*.csv")):
                if csv not in seen_csvs:
                    seen_csvs.add(csv)
                    _print_csv(csv)

        if state == JobState.COMPLETED:
            ui.success(f"Job {job_id} complete. {len(seen_csvs)} CSV(s) produced.")
        else:
            ui.error(f"Job {job_id} ended with state: {state.value} — check log: {log}")

    def _flow_probe(self):
        ui.banner("Probe: Synthesis + Reward Inspection")
        ui.info("Select a stencil, review its config, optionally override params,")
        ui.info("then run synthesis and inspect the reward breakdown.")
        print()

        # 1. Select stencil
        stencils = discover_stencils(self.codegen_apps)
        if not stencils:
            ui.warn("No stencil projects found under " + str(self.codegen_apps))
            return
        stencil = ui.menu("Choose a stencil", [(s.label, s) for s in stencils])
        if stencil is None:
            return
        ui.success("Selected: {}/{}".format(stencil.app_name, stencil.board))

        # 2. Load on-disk config
        if not stencil.config_file or not stencil.config_file.exists():
            ui.error("No config file found for this stencil.")
            return
        with open(stencil.config_file) as f:
            disk_config = json.load(f)

        # Params we allow overriding (the ones probe.py exposes)
        _PROBE_PARAMS = [
            ("vector_factor",               "--vf"),
            ("mem_vector_factor",            "--mvf"),
            ("iter_par_factor",              "--ipf"),
            ("maxi_depth",                   "--depth"),
            ("maxi_read_burst_length",        "--rb"),
            ("maxi_write_burst_length",       "--wb"),
            ("num_read_outstanding",          "--ro"),
            ("num_write_outstanding",         "--wo"),
            ("axis_interconnect_buff_size",   "--axis"),
            ("hls_interconnect_buff_size",    "--hls"),
        ]

        # 3. Show current values and collect overrides
        ui.section("Current config  (press Enter to keep, type new value to override)")
        overrides = {}  # flag -> value (always pass all params so probe.py doesn't fall back to its own defaults)
        for param, flag in _PROBE_PARAMS:
            current = disk_config.get(param, "(not set)")
            raw = ui.prompt(f"{param}", default=str(current))
            try:
                new_val = int(raw)
            except ValueError:
                new_val = current
            overrides[flag] = str(new_val)

        # 4. Dry-run option
        dry_run = not ui.confirm("Run full synthesis (takes several minutes)?")

        # 5. Timeout
        timeout_default = "3600"
        if not dry_run:
            try:
                timeout = int(ui.prompt("Synthesis timeout (seconds)", default=timeout_default))
            except ValueError:
                timeout = int(timeout_default)
        else:
            timeout = int(timeout_default)

        # 6. Build probe.py command
        probe_py = _PROJECT_FILES_DIR / "trainer" / "probe.py"
        cmd = [sys.executable, str(probe_py), stencil.app_name]
        for flag, val in overrides.items():
            cmd += [flag, val]
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd += ["--timeout", str(timeout)]

        ui.section("Running probe")
        ui.info("Command: " + " ".join(cmd))
        print()

        # 7. Run, streaming output directly to the terminal
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            ui.error("probe.py not found at: " + str(probe_py))
        except KeyboardInterrupt:
            print()
            ui.warn("Probe interrupted.")

    def _flow_train_multinode(self):
        ui.banner("Multi-Node DQN Training (parallel SLURM workers)")
        ui.info("Submits one trainer job (normal partition) + N worker array jobs (fpgasynthesis).")
        ui.info("Workers run synthesis and write to a shared SQLite experience store.")
        ui.info("The trainer reads from the store and performs gradient updates continuously.")
        print()

        try:   num_workers = int(ui.prompt("Number of worker nodes", default="4"))
        except ValueError: num_workers = 4

        try:   worker_batch = int(ui.prompt("Parallel synthesis jobs per worker", default="4"))
        except ValueError: worker_batch = 4

        try:   trainer_time = float(ui.prompt("Trainer wall time (hours)", default="12.0"))
        except ValueError: trainer_time = 12.0

        try:   worker_time = float(ui.prompt("Worker wall time (hours)", default="6.0"))
        except ValueError: worker_time = 6.0

        load_model = None
        reset_epsilon = None
        pretrain_stores = None
        pretrain_epochs = 3
        if ui.confirm("Resume trainer from an existing model?", default=False):
            if self._selected_model and ui.confirm(
                    "Use selected model ({})?  ".format(self._selected_model.path.name)):
                load_model = str(self._selected_model.path)
            if load_model is None:
                load_model = ui.prompt("Path to .pth file") or None
            if load_model:
                if ui.confirm("Reset epsilon for fresh exploration?", default=True):
                    try:
                        reset_epsilon = float(ui.prompt("Epsilon value", default="0.3"))
                    except ValueError:
                        reset_epsilon = 0.3

        if ui.confirm("Pre-train on existing experience store(s)?", default=False):
            ui.info("Enter paths to experience_store.db file(s), one per line. Blank line to finish.")
            stores = []
            while True:
                p = ui.prompt("Store path (blank to finish)") or ""
                if not p:
                    break
                stores.append(str(Path(p).resolve()))
            if stores:
                pretrain_stores = stores
                try:
                    pretrain_epochs = int(ui.prompt("Pre-training epochs", default="3"))
                except ValueError:
                    pretrain_epochs = 3

        config = MultiNodeConfig(
            num_workers=num_workers,
            worker_batch_size=worker_batch,
            trainer_time_h=trainer_time,
            worker_time_h=worker_time,
            load_model=load_model,
            reset_epsilon=reset_epsilon,
            pretrain_stores=pretrain_stores,
            pretrain_epochs=pretrain_epochs,
        )
        tag = time.strftime("%Y%m%d_%H%M%S")
        try:
            result = self.mn_launcher.launch(config, tag=tag)
            ui.success("Multi-node jobs submitted.")
            ui.info("Trainer job : " + result["trainer_job_id"])
            ui.info("Worker array: " + result["worker_job_id"])
            ui.info("Run dir     : " + result["run_dir"])
            ui.info("Store       : " + result["store_path"])
            ui.info("Checkpoint  : " + result["checkpoint_path"])
        except Exception as e:
            ui.error("Failed to submit: " + str(e))

    # ------------------------------------------------------------------
    # Evaluate model across all stencils
    # ------------------------------------------------------------------

    def _flow_evaluate_all(self):
        ui.banner("Evaluate Model Across All Stencils")
        ui.info("Gets rank-1 recommendation for each baseline stencil and compares")
        ui.info("Q-values / rewards against the known-good baseline configs.")
        print()

        # Known-good baseline configs (the DRL-relevant parameters only)
        _BASELINES = {
            "heat3d": {
                "vector_factor": 8,  "mem_vector_factor": 16, "iter_par_factor": 16,
                "maxi_depth": 4096, "maxi_read_burst_length": 32, "maxi_write_burst_length": 32,
                "num_read_outstanding": 4, "num_write_outstanding": 4,
                "axis_interconnect_buff_size": 1024, "hls_interconnect_buff_size": 64,
            },
            "jacobian2d": {
                "vector_factor": 8,  "mem_vector_factor": 16, "iter_par_factor": 7,
                "maxi_depth": 4096, "maxi_read_burst_length": 64, "maxi_write_burst_length": 64,
                "num_read_outstanding": 4, "num_write_outstanding": 4,
                "axis_interconnect_buff_size": 4096, "hls_interconnect_buff_size": 360,
            },
            "jacobian3d": {
                "vector_factor": 8,  "mem_vector_factor": 16, "iter_par_factor": 9,
                "maxi_depth": 4096, "maxi_read_burst_length": 64, "maxi_write_burst_length": 64,
                "num_read_outstanding": 4, "num_write_outstanding": 4,
                "axis_interconnect_buff_size": 1024, "hls_interconnect_buff_size": 10,
            },
            "laplace2d": {
                "vector_factor": 8,  "mem_vector_factor": 16, "iter_par_factor": 30,
                "maxi_depth": 8192, "maxi_read_burst_length": 64, "maxi_write_burst_length": 64,
                "num_read_outstanding": 4, "num_write_outstanding": 4,
                "axis_interconnect_buff_size": 4096, "hls_interconnect_buff_size": 50,
            },
            "poisson2d": {
                "vector_factor": 8,  "mem_vector_factor": 16, "iter_par_factor": 27,
                "maxi_depth": 8192, "maxi_read_burst_length": 64, "maxi_write_burst_length": 64,
                "num_read_outstanding": 8, "num_write_outstanding": 8,
                "axis_interconnect_buff_size": 8192, "hls_interconnect_buff_size": 10,
            },
        }

        # 1. Model
        if self._selected_model is None:
            self._flow_select_model()
            if self._selected_model is None:
                return
        model = self._selected_model
        ui.info("Model: " + str(model.path))
        print()

        # 2. Strategy
        strategy = ui.menu(
            "Recommendation strategy",
            [
                ("Balanced  - equal weight on resources & performance", "balanced"),
                ("Performance - maximise frequency, minimise latency",   "performance"),
                ("Resource  - minimise BRAM / DSP / LUT / FF",          "resource"),
            ],
        )
        if strategy is None:
            return

        # 3. Discover stencils — only the 5 baseline apps
        all_stencils = [s for s in discover_stencils(self.codegen_apps) if s.board == "u280"]
        stencils = [s for s in all_stencils if s.app_name in _BASELINES]
        if not stencils:
            ui.error("No baseline stencil projects found under " + str(self.codegen_apps))
            return
        # Sort in consistent order
        stencils.sort(key=lambda s: list(_BASELINES.keys()).index(s.app_name))
        ui.info(f"Found {len(stencils)}/{len(_BASELINES)} baseline stencil(s)")
        print()

        _PROBE_FLAGS = [
            ("vector_factor",              "--vf"),
            ("mem_vector_factor",           "--mvf"),
            ("iter_par_factor",             "--ipf"),
            ("maxi_depth",                  "--depth"),
            ("maxi_read_burst_length",       "--rb"),
            ("maxi_write_burst_length",      "--wb"),
            ("num_read_outstanding",         "--ro"),
            ("num_write_outstanding",        "--wo"),
            ("axis_interconnect_buff_size",  "--axis"),
            ("hls_interconnect_buff_size",   "--hls"),
        ]
        probe_py = _PROJECT_FILES_DIR / "trainer" / "probe.py"

        def _run_probe(app_name, config):
            """Run probe.py, print output, and return the synthesis mean reward or None."""
            import re as _re
            cmd = [sys.executable, str(probe_py), app_name]
            for param, flag in _PROBE_FLAGS:
                if param in config:
                    cmd += [flag, str(config[param])]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
            m = _re.search(r'total \(mean\)\s*=\s*([0-9]+\.[0-9]+)', result.stdout)
            return float(m.group(1)) if m else None

        results = []
        for stencil in stencils:
            ui.section(f"{stencil.app_name}")

            # Use hardcoded baseline config
            baseline_config = _BASELINES[stencil.app_name]

            # Get rank-1 recommendation
            try:
                recs = self.recommender.run_to_json(
                    stencil, model.path,
                    output_json=None,
                    top_k=1,
                    strategy=strategy,
                )
            except Exception as e:
                ui.error(f"  Recommender failed: {e}")
                results.append({"app": stencil.app_name, "error": str(e)})
                continue

            rec_config   = recs[0]["config"]
            rec_qvals    = recs[0]["q_values"]   # list [lat, bram, dsp, lut, ff, freq]
            rec_score    = recs[0].get("strategy_score", 0.0)

            # Score the baseline config through the same model
            try:
                baseline_score = self.recommender.score_config(
                    baseline_config, stencil, model.path, strategy=strategy
                )
            except Exception:
                baseline_score = None

            score_ratio = (rec_score / baseline_score) if baseline_score else None

            print(f"  Baseline score : {baseline_score:.4f}" if baseline_score is not None else "  Baseline score : ?")
            print(f"  Rec score      : {rec_score:.4f}")
            print(f"  Ratio          : {score_ratio:.3f}" if score_ratio is not None else "  Ratio          : ?")
            print(f"  Q-values (lat={rec_qvals[0]:.3f} bram={rec_qvals[1]:.3f} "
                  f"dsp={rec_qvals[2]:.3f} lut={rec_qvals[3]:.3f} "
                  f"ff={rec_qvals[4]:.3f} freq={rec_qvals[5]:.3f})")

            row = {
                "app":            stencil.app_name,
                "baseline_score": baseline_score,
                "rec_score":      rec_score,
                "score_ratio":    score_ratio,
                "rec_config":     rec_config,
                "baseline_config": baseline_config,
                "q_values":       dict(zip(["runtime","bram","dsp","lut","ff","freq"], rec_qvals)),
            }

            print()
            ui.info("  → Probing recommended config...")
            synth_rec = _run_probe(stencil.app_name, rec_config)
            if synth_rec is not None:
                row["rec_score"] = synth_rec

            print()
            ui.info("  → Probing baseline config...")
            synth_bl = _run_probe(stencil.app_name, baseline_config)
            if synth_bl is not None:
                row["baseline_score"] = synth_bl

            if row["baseline_score"] is not None and row["rec_score"] is not None:
                row["score_ratio"] = row["rec_score"] / row["baseline_score"] if row["baseline_score"] else None

            results.append(row)
            print()

        # Summary table
        ui.banner("Evaluation Summary")
        header = f"  {'App':<20}  {'BL Score':>9}  {'Rec Score':>9}  {'Ratio':>6}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for r in results:
            if "error" in r:
                print(f"  {r['app']:<20}  {'ERROR':>9}  {r['error'][:30]}")
                continue
            bl_str    = f"{r['baseline_score']:.4f}" if r["baseline_score"] is not None else "?"
            rec_str   = f"{r['rec_score']:.4f}"
            ratio_str = f"{r['score_ratio']:.3f}"  if r["score_ratio"]    is not None else "?"
            print(
                f"  {r['app']:<20}  {bl_str:>9}  {rec_str:>9}  {ratio_str:>6}"
            )
        print()

        # Save results to JSON
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _PROJECT_FILES_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"evaluate_all_{ts}_{strategy}.json"
        output = {
            "timestamp": ts,
            "model": str(model.path),
            "strategy": strategy,
            "results": [
                {
                    "app":             r["app"],
                    "baseline_score":  r.get("baseline_score"),
                    "rec_score":       r.get("rec_score"),
                    "score_ratio":     r.get("score_ratio"),
                    "beats_baseline":  (r.get("score_ratio") or 0) >= 1.0,
                    "q_values":        r.get("q_values"),
                    "recommended_config":  r.get("rec_config"),
                    "baseline_config":     r.get("baseline_config"),
                }
                for r in results if "error" not in r
            ],
        }
        import json as _json
        with open(out_path, "w") as f:
            _json.dump(output, f, indent=2)
        ui.info(f"Results saved to: {out_path}")
        print()

    def _flow_validate_all(self):
        ui.banner("Validate All: Submit HW Builds from Evaluation Results")
        ui.info("Loads a saved evaluate_all JSON, applies the recommended config")
        ui.info("to each stencil project, and submits a full HW build SLURM job.")
        print()

        # Find available evaluate_all result files
        results_dir = _PROJECT_FILES_DIR / "results"
        result_files = sorted(results_dir.glob("evaluate_all_*.json"), reverse=True) if results_dir.exists() else []

        if not result_files:
            ui.warn("No evaluate_all result files found in: " + str(results_dir))
            ui.info("Run 'Evaluate model across all stencils' first to generate results.")
            return

        chosen_file = ui.menu(
            "Choose evaluation results to build from",
            [(f.name, f) for f in result_files],
        )
        if chosen_file is None:
            return

        with open(chosen_file) as f:
            eval_data = json.load(f)

        ui.info(f"Model    : {eval_data.get('model', '?')}")
        ui.info(f"Strategy : {eval_data.get('strategy', '?')}")
        ui.info(f"Timestamp: {eval_data.get('timestamp', '?')}")
        print()

        entries = eval_data.get("results", [])
        if not entries:
            ui.warn("No results found in the selected file.")
            return

        # Show what will be built
        ui.section("Recommended configs to build")
        for e in entries:
            flag = "✔" if e.get("beats_baseline") else "✘"
            bl  = f"{e['baseline_score']:.4f}" if e.get("baseline_score") is not None else "?"
            rec = f"{e['rec_score']:.4f}"      if e.get("rec_score")      is not None else "?"
            print(f"  {flag}  {e['app']:<20}  BL={bl}  Rec={rec}")
        print()

        if not ui.confirm(f"Submit {len(entries)} HW build job(s)?", default=True):
            return

        # Map app_name -> StencilProject for u280 board
        all_stencils = {s.app_name: s for s in discover_stencils(self.codegen_apps) if s.board == "u280"}

        submitted = []
        failed    = []
        for e in entries:
            app = e["app"]
            rec_config = e.get("recommended_config")
            if not rec_config:
                ui.warn(f"  {app}: no recommended_config in results — skipping")
                failed.append({"app": app, "reason": "no config in results file"})
                continue

            stencil = all_stencils.get(app)
            if stencil is None:
                ui.warn(f"  {app}: stencil project not found — skipping")
                failed.append({"app": app, "reason": "stencil project not found"})
                continue

            try:
                apply_config_to_project(rec_config, stencil)
                ui.success(f"  {app}: config applied")
            except Exception as exc:
                ui.error(f"  {app}: failed to apply config — {exc}")
                failed.append({"app": app, "reason": f"apply_config failed: {exc}"})
                continue

            try:
                job = self.builder.submit_build(stencil)
                ui.success(f"  {app}: build job {job.job_id} submitted  (log: {job.log_path.name})")
                submitted.append({"app": app, "job_id": job.job_id, "log": str(job.log_path)})
            except Exception as exc:
                ui.error(f"  {app}: sbatch failed — {exc}")
                failed.append({"app": app, "reason": f"sbatch failed: {exc}"})

        print()
        ui.section("Summary")
        print(f"  Submitted : {len(submitted)}")
        print(f"  Failed    : {len(failed)}")
        if submitted:
            print()
            print(self.builder.squeue_summary())

        # Save a record alongside the source eval file
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = results_dir / f"validate_all_{ts}.json"
        with open(log_path, "w") as f:
            json.dump({
                "timestamp":    ts,
                "source_file":  str(chosen_file),
                "model":        eval_data.get("model"),
                "strategy":     eval_data.get("strategy"),
                "submitted":    submitted,
                "failed":       failed,
            }, f, indent=2)
        ui.info(f"Job log saved to: {log_path}")
        print()

    def _flow_train(self):
        ui.banner("Train New DQN Model")

        training_style = ui.menu(
            "Training mode",
            [
                ("Single-node (this machine / one SLURM job)",  "single"),
                ("Multi-node  (trainer + parallel worker jobs)", "multi"),
            ],
        )
        if training_style is None:
            return
        if training_style == "multi":
            self._flow_train_multinode()
            return

        mode = ui.menu(
            "Training budget",
            [
                ("Fixed number of synthesis runs", "runs"),
                ("Fixed wall-clock time (hours)",  "time"),
                ("Both (whichever comes first)",   "both"),
            ],
        )
        if mode is None:
            return

        num_runs = None
        max_time = None
        if mode in ("runs", "both"):
            try:   num_runs = int(ui.prompt("Number of synthesis runs", default="200"))
            except ValueError: num_runs = 200
        if mode in ("time", "both"):
            try:   max_time = float(ui.prompt("Max training time (hours)", default="2.0"))
            except ValueError: max_time = 2.0

        try:   batch_size = int(ui.prompt("Parallel synthesis jobs", default="4"))
        except ValueError: batch_size = 4

        load_model = None
        if ui.confirm("Resume from an existing model?", default=False):
            if self._selected_model and ui.confirm(
                    "Use selected model ({})?".format(self._selected_model.path.name)):
                load_model = str(self._selected_model.path)
            if load_model is None:
                load_model = ui.prompt("Path to .pth file") or None

        config = TrainConfig(
            num_runs=num_runs, max_time_h=max_time,
            batch_size=batch_size, load_model=load_model,
        )
        tag = time.strftime("%Y%m%d_%H%M%S")
        try:
            job = self.trainer.launch(config, tag=tag)
            ui.success("Training job submitted: " + job.job_id)
            ui.info("Log: " + str(job.log_path))
        except Exception as e:
            ui.error("Failed to submit: " + str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HLS Stencil Optimiser")
    parser.add_argument("--root", default=None, help="Override project root directory")
    args = parser.parse_args()

    root = Path(args.root) if args.root else _PROJECT_ROOT
    if not root.exists():
        print("Error: project root not found: " + str(root), file=sys.stderr)
        sys.exit(1)

    try:
        Orchestrator(root).run()
    except KeyboardInterrupt:
        print("\nInterrupted.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()

# --- repo-root bootstrap (added when this entry point moved into scripts/) ---
# Every data path in this file is relative to the repository root, so run it from there:
#     cd ~/Projects/SDF_3 && python scripts/main.py
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---

import os
import re
import sys
import shutil
import logging
import questionary
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from loguru import logger
from typing import Optional

from src.DataPreprocessor import DataPreprocessor

# JAX/XLA memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

# Suppress noisy ABSL and TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

class SDFApp:
    """
    UI and workspace management for SDF Tools v3.
    """

    def __init__(self):
        self._work_folder: Optional[Path] = None
        self._log_format = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"

    # ===== Private APIs =====

    def _setup_console_logger(self) -> None:
        """
        Sets up the default console logging sink.
        """
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), colorize=True, format=self._log_format, level="INFO")

    def _ui_init(self) -> Optional[str]:
        """
        Presents the initialization / directory selection menu to the user.
        """
        choice = questionary.select(
            "Select your working run folder:",
            choices=[
                "Initialise a new run folder",
                "Select an existing run folder",
                "Exit"
            ],
            instruction="(Use arrow keys)",
        ).ask()
        return choice

    def _ui_select_working_folder(self) -> Optional[Path]:
        """
        Presents a menu to choose an existing run folder from data/runs/.
        """
        runs_root = Path("data/runs")
        if not runs_root.exists() or not any(runs_root.iterdir()):
            logger.error("'data/runs' does not exist, you need to initialise first.")
            return None
            
        folders = [f.name for f in runs_root.iterdir() if f.is_dir()]
        folders.sort(reverse=True)
        if not folders:
            logger.error("No run folder found.")
            return None
            
        selected_folder = questionary.select(
            "Please select a run folder:",
            choices=folders,
            instruction="(Use arrow keys to choose)"
        ).ask()
        
        if selected_folder:
            target_path = runs_root / selected_folder
            logger.success(f"Working run folder set to {selected_folder}.")
            return target_path
        return None

    def _ui_main(self) -> Optional[str]:
        """
        Presents the main task selection menu (Data Preprocessing and Training for v3).
        """
        choice = questionary.select(
            "Select your task:",
            choices=[
                "1.1 Data Preprocessing (Standard)",
                "1.2 Data Preprocessing (Wall Variants)",
                "2.  Train Embeddings & Write Back to IFC",
                "2.2 Train Embeddings (HashGrid, experimental) & Write Back to IFC",
                "2.3 HashGrid TEST: single IFC -> train -> reconstruct meshes",
                "3.1 Generate Spatial Graph (Standard)",
                "3.2 Generate Combinatorial Graphs (Wall Variants)",
                "4.  Train Graphormer & Full Test Evaluation",
                "5.  Full Experiment (1.2 -> 2 -> 3.2 -> 4 + Holdout)",
                "5.2 Full Experiment HashGrid (1.2 -> 2.2 -> 3.2 -> 4 + Holdout + Archive)",
                "6.  Re-train Graphormer on existing graphs (reuse embedding; head_mode from config)",
                "6.2 Re-train GIN BASELINE on existing graphs (message-passing; forces model_arch=gin)",
                "7.  Evaluate ONLY (existing checkpoint: test + holdout + archive, no training)",
                "8.  Ablation: single-rule balanced datasets (3 runs: six_window / single_wall / semantic)",
                "8.2 Ablation: ONE rule only (pick rule; same balanced subset as menu 8)",
                "0.  Exit"
            ],
            instruction=f"\n Current run folder {self._work_folder} \n (Use arrow keys)",
        ).ask()
        return choice

    def _initialise_project(self) -> Path:
        """
        Initialises a brand new run folder with subfolders and copies over the entire input directory.
        """
        # ==== Create project folder =====
        Path("data/runs").mkdir(exist_ok=True)
        new_folder_name = datetime.now().strftime("%Y%b%d%H%M").upper()
        new_folder_path = Path("data/runs/") / new_folder_name
        new_folder_path.mkdir(exist_ok=True)
        logger.success("New run folder created")

        # ==== Create sub folders =====
        (new_folder_path / "input").mkdir(parents=True, exist_ok=True)
        (new_folder_path / "intermediate" / "expanded_ifcs").mkdir(parents=True, exist_ok=True)
        (new_folder_path / "intermediate" / "embedded_ifcs").mkdir(parents=True, exist_ok=True)
        (new_folder_path / "intermediate" / "objects").mkdir(parents=True, exist_ok=True)
        (new_folder_path / "output").mkdir(parents=True, exist_ok=True)
        logger.success("Sub folders created")

        # ==== Copy files =====
        input_source = Path("data/input")
        if not input_source.exists():
            logger.error(f"Source input directory not found: {input_source}")
            return new_folder_path
            
        shutil.copytree(input_source, new_folder_path / "input", dirs_exist_ok=True)
        logger.success("Input files copied to new run folder")
        logger.success(f"Working run folder set to {new_folder_path}.")
        logger.info("Please adjust your configs in config.yaml under input/input_config/")
        return new_folder_path

    # ===== Public APIs =====

    @logger.catch
    def run(self) -> None:
        """
        Starts the main application event loop.
        """
        self._setup_console_logger()
        logger.info("========================================")
        logger.info("          SDF tools v3                ")
        logger.info("========================================")
        
        while True:
            choice_init = self._ui_init()

            if choice_init is None or "Exit" in choice_init:
                logger.info("Exiting...")
                break

            if "Initialise" in choice_init:
                self._work_folder = self._initialise_project()
                logger.info("-" * 40)

            elif "Select" in choice_init:
                self._work_folder = self._ui_select_working_folder()
                logger.info("-" * 40)

            if self._work_folder:
                while True:
                    choice_task = self._ui_main()
                    if choice_task is None:
                        logger.info("Exiting...")
                        sys.exit()

                    parts = choice_task.split()
                    code = parts[0] if parts else ""
                    cfg_path = self._work_folder / "input" / "input_config" / "config.yaml"
                    wf = self._work_folder

                    if code == "0.":
                        logger.info("Exiting...")
                        sys.exit()

                    elif code == "1.1":
                        self._with_stage_logging(
                            "preprocessing.log", "Data Preprocessing (Standard)",
                            lambda: DataPreprocessor.execute(cfg_path, wf)
                        )

                    elif code == "1.2":
                        def _variant_preprocess():
                            logger.info("=== Step 1/2: Generating Wall Length Variants IFC ===")
                            DataPreprocessor.generate_wall_variants_ifc(wf)
                            logger.info("=== Step 2/2: Preprocessing with forced rotation expansion ===")
                            processor = DataPreprocessor.config_load(cfg_path)
                            processor.config.extraction.data_expand = True
                            processor._run(wf)
                            del processor
                        self._with_stage_logging(
                            "wall_variants_preprocessing.log",
                            "Data Preprocessing (Wall Variants)", _variant_preprocess
                        )

                    elif code == "2.":
                        def _train_embeddings():
                            from src.Embedding import SDFEmbeddingTrainer
                            SDFEmbeddingTrainer.execute(cfg_path, wf)
                        self._with_stage_logging(
                            "training.log", "Train Embeddings & Write Back to IFC",
                            _train_embeddings
                        )

                    elif code == "2.2":
                        def _train_embeddings_hash():
                            from src.EmbeddingHash import SDFHashEmbeddingTrainer
                            SDFHashEmbeddingTrainer.execute(cfg_path, wf)
                        self._with_stage_logging(
                            "training_hash.log", "Train Embeddings (HashGrid) & Write Back to IFC",
                            _train_embeddings_hash
                        )

                    elif code == "2.3":
                        def _hash_single_ifc_test():
                            from src.EmbeddingHash import SDFHashEmbeddingTrainer
                            SDFHashEmbeddingTrainer.run_single_ifc_test(cfg_path, wf)
                        self._with_stage_logging(
                            "hash_single_ifc_test.log",
                            "HashGrid TEST: single IFC -> train -> reconstruct", _hash_single_ifc_test
                        )

                    elif code == "3.1":
                        self._with_stage_logging(
                            "graph_generation.log", "Generate Spatial Graph (Standard)",
                            lambda: DataPreprocessor.generate_graphs(cfg_path, wf)
                        )

                    elif code == "3.2":
                        self._with_stage_logging(
                            "combinatorial_graph_generation.log",
                            "Generate Combinatorial Graphs (Wall Variants)",
                            lambda: DataPreprocessor.generate_combinatorial_graphs(cfg_path, wf)
                        )

                    elif code == "4.":
                        def _train_and_eval():
                            from src.Graphormer import SDFGraphormerTrainer
                            logger.info("=== Step 1/3: Training Graphormer (60/20/20 split) ===")
                            SDFGraphormerTrainer.execute_with_split(cfg_path, wf)
                            logger.info("=== Step 2/3: Full Test-Set Evaluation ===")
                            SDFGraphormerTrainer.evaluate_test_set(cfg_path, wf)
                            logger.info("=== Step 3/3: Held-out (Unseen-Transform) Evaluation ===")
                            SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, wf)
                            self._log_seen_vs_holdout(wf)
                        self._with_stage_logging(
                            "graphormer_train_and_eval.log",
                            "Train Graphormer & Full Test Evaluation", _train_and_eval
                        )

                    elif code == "5.":
                        def _full_experiment():
                            from src.Embedding import SDFEmbeddingTrainer
                            from src.Graphormer import SDFGraphormerTrainer
                            logger.info("=== [1/5] Data Preprocessing (Wall Variants: generate + rotate/translate) ===")
                            DataPreprocessor.generate_wall_variants_ifc(wf)
                            processor = DataPreprocessor.config_load(cfg_path)
                            processor.config.extraction.data_expand = True
                            processor._run(wf)
                            del processor
                            logger.info("=== [2/5] Train Embeddings & Write Back to IFC ===")
                            SDFEmbeddingTrainer.execute(cfg_path, wf)
                            logger.info("=== [3/5] Generate Combinatorial Graphs (+ isolate holdout copies) ===")
                            DataPreprocessor.generate_combinatorial_graphs(cfg_path, wf)
                            logger.info("=== [4/5] Train Graphormer + Full Test Evaluation ===")
                            SDFGraphormerTrainer.execute_with_split(cfg_path, wf)
                            SDFGraphormerTrainer.evaluate_test_set(cfg_path, wf)
                            logger.info("=== [5/5] Held-out (Unseen-Transform) Evaluation ===")
                            SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, wf)
                            self._log_seen_vs_holdout(wf)
                        self._with_stage_logging(
                            "full_experiment.log",
                            "Full Experiment (1.2 -> 2 -> 3.2 -> 4 + Holdout)", _full_experiment
                        )

                    elif code == "5.2":
                        def _full_experiment_hash():
                            from src.EmbeddingHash import SDFHashEmbeddingTrainer
                            from src.Graphormer import SDFGraphormerTrainer
                            logger.info("=== [1/5] Data Preprocessing (Wall Variants: generate + rotate/translate) ===")
                            DataPreprocessor.generate_wall_variants_ifc(wf)
                            processor = DataPreprocessor.config_load(cfg_path)
                            processor.config.extraction.data_expand = True
                            processor._run(wf)
                            del processor
                            logger.info("=== [2/5] Train Embeddings (HashGrid) & Write Back to IFC ===")
                            SDFHashEmbeddingTrainer.execute(cfg_path, wf)
                            logger.info("=== [3/5] Generate Combinatorial Graphs (+ isolate holdout copies) ===")
                            DataPreprocessor.generate_combinatorial_graphs(cfg_path, wf)
                            logger.info("=== [4/5] Train Graphormer + Full Test Evaluation ===")
                            SDFGraphormerTrainer.execute_with_split(cfg_path, wf)
                            SDFGraphormerTrainer.evaluate_test_set(cfg_path, wf)
                            logger.info("=== [5/5] Held-out (Unseen-Transform) Evaluation ===")
                            SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, wf)
                            self._log_seen_vs_holdout(wf)
                            self._archive_experiment(wf, cfg_path)
                        self._with_stage_logging(
                            "full_experiment_hash.log",
                            "Full Experiment HashGrid (1.2 -> 2.2 -> 3.2 -> 4 + Holdout)", _full_experiment_hash
                        )

                    elif code == "6.":
                        def _retrain_existing():
                            from src.Graphormer import SDFGraphormerTrainer
                            logger.info("=== [1/3] Train Graphormer on EXISTING graphs (reusing embedding) ===")
                            SDFGraphormerTrainer.execute_with_split(cfg_path, wf)
                            logger.info("=== [2/3] Full Test-Set Evaluation ===")
                            SDFGraphormerTrainer.evaluate_test_set(cfg_path, wf)
                            logger.info("=== [3/3] Held-out (Unseen-Transform) Evaluation ===")
                            SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, wf)
                            self._log_seen_vs_holdout(wf)
                            self._archive_experiment(wf, cfg_path)
                        self._with_stage_logging(
                            "graphormer_retrain.log",
                            "Re-train Graphormer on existing graphs", _retrain_existing
                        )

                    elif code == "6.2":
                        def _retrain_gin_baseline():
                            from src.Graphormer import SDFGraphormerTrainer
                            logger.info("=== [1/3] Train GIN baseline on EXISTING graphs (reusing embedding) ===")
                            SDFGraphormerTrainer.execute_with_split(cfg_path, wf, model_arch="gin")
                            logger.info("=== [2/3] Full Test-Set Evaluation ===")
                            SDFGraphormerTrainer.evaluate_test_set(cfg_path, wf, model_arch="gin")
                            logger.info("=== [3/3] Held-out (Unseen-Transform) Evaluation ===")
                            SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, wf, model_arch="gin")
                            self._log_seen_vs_holdout(wf)
                            self._archive_experiment(wf, cfg_path, arch_override="gin")
                        self._with_stage_logging(
                            "gin_baseline_retrain.log",
                            "Re-train GIN baseline on existing graphs", _retrain_gin_baseline
                        )

                    elif code == "7.":
                        def _evaluate_only():
                            from src.Graphormer import SDFGraphormerTrainer
                            logger.info("=== [1/2] Full Test-Set Evaluation (existing checkpoint) ===")
                            SDFGraphormerTrainer.evaluate_test_set(cfg_path, wf)
                            logger.info("=== [2/2] Held-out (Unseen-Transform) Evaluation ===")
                            SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, wf)
                            self._log_seen_vs_holdout(wf)
                            self._archive_experiment(wf, cfg_path)
                        self._with_stage_logging(
                            "evaluation_only.log",
                            "Evaluate ONLY (existing checkpoint)", _evaluate_only
                        )

                    elif code == "8.":
                        self._with_stage_logging(
                            "single_rule_ablation.log",
                            "Single-Rule Ablation (six_window / single_wall / semantic)",
                            lambda: self._run_single_rule_ablation(wf, cfg_path)
                        )

                    elif code == "8.2":
                        rule = questionary.select(
                            "Which rule to ablate?",
                            choices=["six_window", "single_wall", "semantic"],
                            instruction="(Use arrow keys)",
                        ).ask()
                        if rule:
                            self._with_stage_logging(
                                f"single_rule_ablation_{rule}.log",
                                f"Single-Rule Ablation ({rule} only)",
                                lambda: self._run_single_rule_ablation(wf, cfg_path, rules=(rule,))
                            )

                    else:
                        logger.warning(f"Unknown selection: {choice_task}")

    def _archive_experiment(self, work_folder: Path, cfg_path: Path, arch_override: Optional[str] = None,
                            dest_name: Optional[str] = None) -> None:
        """
        Copies the eval reports + split metrics into output/experiments/<head_mode>/ so that
        running several head modes on the same graphs does not clobber each other's results.
        `arch_override` must match any model_arch override passed to the train/eval calls,
        so the archive folder is labeled with the architecture that actually ran.
        `dest_name` overrides the derived folder name entirely (used by the ablation runs).
        """
        import yaml
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            g = (cfg.get("training", {}) or {}).get("graphormer", {}) or {}
            mode = g.get("head_mode") or ("offset" if g.get("residual_anchored", True) else "absolute")
            pose_w = float(g.get("pose_distill_weight") or 0.0)
            if pose_w > 0.0:
                mode = f"{mode}_pose_w{pose_w:g}"
            arch = str(arch_override or g.get("model_arch") or "graphormer").strip().lower()
            if arch != "graphormer":
                mode = f"{arch}_{mode}"
        except Exception:
            mode = "unknown"
        if dest_name:
            mode = dest_name
        dst = work_folder / "output" / "experiments" / str(mode)
        dst.mkdir(parents=True, exist_ok=True)
        eval_dir = work_folder / "output" / "evaluation_results"
        for name in ("test_evaluation_report.json", "holdout_evaluation_report.json"):
            src = eval_dir / name
            if src.exists():
                shutil.copy2(src, dst / name)
        metrics = work_folder / "output" / "graphormer_split_metrics.json"
        if metrics.exists():
            shutil.copy2(metrics, dst / "graphormer_split_metrics.json")
        logger.success(f"Archived '{mode}' experiment results to: {dst}")

    @staticmethod
    def _ablation_rule_of(filename: str) -> Optional[str]:
        """
        Maps a combinatorial graph filename to the rule that placed its door,
        mirroring determine_door_position in src/DataPreprocessor.py
        (effective wall = present AND length > 1000 mm):
          six_window  - neither changing wall effective: door at the complement slot of the TYPE2 window
          single_wall - exactly one effective wall: door on that side
          semantic    - both walls effective: toilet side decides
        Returns None for filenames that don't match the combinatorial pattern.
        """
        m = re.match(r"graph_room\d+_aug\d+_L(Absent|\d+)_R(Absent|\d+)_W\d+_graph\.json$", filename)
        if not m:
            return None
        left_eff = m.group(1) != "Absent" and int(m.group(1)) > 1000
        right_eff = m.group(2) != "Absent" and int(m.group(2)) > 1000
        if not left_eff and not right_eff:
            return "six_window"
        if left_eff != right_eff:
            return "single_wall"
        return "semantic"

    def _run_single_rule_ablation(self, wf: Path, cfg_path: Path,
                                  rules: Optional[tuple] = None) -> None:
        """
        Single-factor ablation on an EXISTING run folder (embeddings + graphs reused):
        splits the combinatorial dataset by placement rule, balances the three training
        sets to the size of the smallest rule (seeded sample), then for each rule runs
        train -> test eval -> holdout eval and archives the results to
        output/experiments/ablation_<rule>/. Holdout sets are per-rule but NOT
        subsampled. All graphs are restored to output/ (+holdout/) afterwards, even on failure.

        `rules` restricts which rules are trained (menu 8.2). Balancing always uses the
        min count across ALL rules and the fixed sampling seed, so a restricted run trains
        on the identical subset a full menu-8 pass would — only config.yaml's training
        seed differentiates replications.
        """
        import json
        import random
        from src.Graphormer import SDFGraphormerTrainer

        RULES = ("six_window", "single_wall", "semantic")
        SEED = 42
        run_rules = tuple(rules) if rules else RULES
        unknown = set(run_rules) - set(RULES)
        if unknown:
            raise ValueError(f"Unknown ablation rule(s): {sorted(unknown)}")
        out_dir = wf / "output"
        hold_dir = out_dir / "holdout"
        pool_root = out_dir / "ablation_pool"
        pool_train = pool_root / "train"
        pool_hold = pool_root / "holdout"
        pool_train.mkdir(parents=True, exist_ok=True)
        pool_hold.mkdir(parents=True, exist_ok=True)
        hold_dir.mkdir(parents=True, exist_ok=True)

        def _move_graphs(src: Path, dst: Path, names: Optional[list] = None) -> int:
            if names is None:
                names = [p.name for p in src.glob("graph_room*.json")]
            for name in names:
                shutil.move(str(src / name), str(dst / name))
            return len(names)

        # Sweep every graph (incl. leftovers of an interrupted ablation) into the pool;
        # the finally block below restores everything from it.
        pooled = _move_graphs(out_dir, pool_train) + _move_graphs(hold_dir, pool_hold)
        logger.info(f"Pooled {pooled} graph files under {pool_root.as_posix()}")

        # Each round wipes the split-checkpoint dir, which would destroy the full-run
        # (all-rules) best checkpoint that menu 7 re-evaluates -- park it aside first.
        ckpt_dir = wf / "intermediate" / "graphormer_jax_split_checkpoints"
        ckpt_backup = wf / "intermediate" / "graphormer_jax_split_checkpoints_pre_ablation"
        if ckpt_dir.exists() and not ckpt_backup.exists():
            shutil.move(str(ckpt_dir), str(ckpt_backup))
            logger.info(f"Existing full-run checkpoint parked at {ckpt_backup.as_posix()}")

        def _classify(folder: Path) -> dict:
            groups = {r: [] for r in RULES}
            unmatched = 0
            for p in sorted(folder.glob("graph_room*.json")):
                rule = self._ablation_rule_of(p.name)
                if rule is None:
                    unmatched += 1
                    continue
                groups[rule].append(p.name)
            if unmatched:
                logger.warning(f"{unmatched} graph files in {folder.as_posix()} did not match "
                               f"the combinatorial filename pattern and are excluded.")
            return groups

        try:
            train_groups = _classify(pool_train)
            hold_groups = _classify(pool_hold)
            counts = {r: len(train_groups[r]) for r in RULES}
            n_balanced = min(counts.values())
            if n_balanced == 0:
                raise RuntimeError(f"At least one rule has no training graphs: {counts}")
            logger.info(f"Training graphs per rule: {counts} -> balanced to {n_balanced} each (seed {SEED})")
            logger.info(f"Holdout graphs per rule (not subsampled): "
                        f"{ {r: len(hold_groups[r]) for r in RULES} }")

            balanced = {
                r: sorted(random.Random(f"{SEED}_{r}").sample(train_groups[r], n_balanced))
                for r in RULES
            }

            for i, rule in enumerate(run_rules, 1):
                logger.info(f"=== [{i}/{len(run_rules)}] Single-rule run '{rule}': "
                            f"{n_balanced} train / {len(hold_groups[rule])} holdout graphs ===")
                _move_graphs(pool_train, out_dir, balanced[rule])
                _move_graphs(pool_hold, hold_dir, hold_groups[rule])

                # Checkpoints are saved as best_<epoch>; a stale one from an earlier round
                # with a higher epoch would win at restore time, so start each round clean.
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)

                SDFGraphormerTrainer.execute_with_split(cfg_path, wf)
                SDFGraphormerTrainer.evaluate_test_set(cfg_path, wf)
                SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, wf)
                self._log_seen_vs_holdout(wf)

                dest_name = f"ablation_{rule}"
                self._archive_experiment(wf, cfg_path, dest_name=dest_name)
                dst = out_dir / "experiments" / dest_name
                try:
                    import yaml
                    training_seed = (yaml.safe_load(cfg_path.read_text())
                                     .get("training", {}).get("reproducibility", {}).get("seed"))
                except Exception:
                    training_seed = None
                manifest = {
                    "rule": rule,
                    "seed": SEED,
                    "training_seed": training_seed,
                    "train_counts_available": counts,
                    "train_size_balanced": n_balanced,
                    "holdout_size": len(hold_groups[rule]),
                    "train_files": balanced[rule],
                    "holdout_files": hold_groups[rule],
                }
                (dst / "ablation_manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8")
                if ckpt_dir.exists():
                    ckpt_dst = dst / "checkpoint"
                    if ckpt_dst.exists():
                        shutil.rmtree(ckpt_dst)
                    shutil.copytree(ckpt_dir, ckpt_dst)
                logger.success(f"Ablation round '{rule}' archived to {dst.as_posix()}")

                # Return this round's files to the pool before staging the next rule.
                _move_graphs(out_dir, pool_train, balanced[rule])
                _move_graphs(hold_dir, pool_hold, hold_groups[rule])
        finally:
            restored = _move_graphs(pool_train, out_dir) + _move_graphs(pool_hold, hold_dir)
            for d in (pool_train, pool_hold, pool_root):
                try:
                    d.rmdir()
                except OSError:
                    pass
            if ckpt_backup.exists():
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                shutil.move(str(ckpt_backup), str(ckpt_dir))
                logger.info("Full-run checkpoint restored to graphormer_jax_split_checkpoints/.")
            logger.info(f"Restored {restored} pooled graph files back to output/ and output/holdout/.")

    def _log_seen_vs_holdout(self, work_folder: Path) -> None:
        """
        Logs a side-by-side comparison of the in-distribution test split vs the unseen
        held-out copies, so the generalization gap under rigid transforms is visible at a glance.
        """
        import json
        eval_dir = work_folder / "output" / "evaluation_results"
        test_p = eval_dir / "test_evaluation_report.json"
        hold_p = eval_dir / "holdout_evaluation_report.json"
        if not test_p.exists():
            return
        test = json.loads(test_p.read_text())["global_metrics"]
        rows = [("Seen test split", test)]
        if hold_p.exists():
            rows.append(("Unseen holdout", json.loads(hold_p.read_text())["global_metrics"]))
        logger.info("================ Seen vs. Held-out ================")
        logger.info(f"{'set':<18}{'mean_euc':>10}{'median':>10}{'rmse':>10}{'L/R acc':>9}{'#doors':>9}")
        for name, m in rows:
            logger.info(
                f"{name:<18}{m['mean_euclidean_error']:>10.4f}{m['median_euclidean_error']:>10.4f}"
                f"{m['rmse_euclidean']:>10.4f}{m.get('left_right_accuracy', float('nan'))*100:>8.1f}%{m['total_doors_evaluated']:>9d}"
            )
        if len(rows) == 2:
            gap = rows[1][1]['mean_euclidean_error'] - rows[0][1]['mean_euclidean_error']
            ratio = rows[1][1]['mean_euclidean_error'] / max(rows[0][1]['mean_euclidean_error'], 1e-9)
            logger.info(f"Generalization gap (holdout - test): {gap:+.4f} m  ({ratio:.2f}x)")
        logger.info("===================================================")

    def _with_stage_logging(self, log_filename: str, title: str, fn) -> None:
        """
        Runs `fn` with DEBUG logs teed to intermediate/<log_filename> and INFO to the
        console, restoring the plain console logger afterwards.
        """
        log_file_path = self._work_folder / "intermediate" / log_filename
        logger.remove()
        logger.add(str(log_file_path), rotation="10 MB", level="DEBUG", encoding="utf-8")
        logger.add(lambda msg: print(msg, end=""), colorize=True, format=self._log_format, level="INFO")

        success = True
        try:
            fn()
        except Exception as e:
            logger.error(f"{title} failed: {e}")
            success = False

        self._setup_console_logger()
        if success:
            logger.success(f"{title} completed. Full logs saved to {log_file_path.as_posix()}")
        logger.info("-" * 40)


if __name__ == "__main__":
    app = SDFApp()
    app.run()
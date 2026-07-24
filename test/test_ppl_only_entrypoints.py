from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_eat_moe_runtime_exposes_skip_eval_flag() -> None:
    text = _read("code/ablation/EAT-MOE/qwen3_eat_moe_ablation.py")

    assert 'parser.add_argument("--skip-eval", action="store_true")' in text
    assert "if not args.skip_eval:" in text


def test_eat_moe_ppl_wrapper_is_ppl_only_and_writes_into_results_subdir() -> None:
    text = _read("code/scripts/EAT-MOE/run_eat_moe_ppl_grid.sh")

    assert 'source "$SCRIPT_DIR/../shared/model_family.sh"' in text
    assert 'OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE_ROOT/results/EAT-MOE/$MODEL_TAG}' in text
    assert "--n-ctx 2048" in text
    assert "--skip-eval" in text


def test_modes_runtime_exposes_skip_mcqa_flag() -> None:
    text = _read("code/ablation/MoDES/run_qwen3_text_pipeline.py")

    assert 'parser.add_argument("--skip_mcqa", action="store_true")' in text
    assert "if not args.skip_mcqa:" in text


def test_modes_ppl_wrapper_is_ppl_only_and_writes_into_results_subdir() -> None:
    text = _read("code/scripts/MoDES/run_modes_qwen3_ppl_pipeline.sh")

    assert 'REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)' in text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE_ROOT/results/MoDES/$MODEL_TAG}' in text
    assert '--model-family "$MODEL_FAMILY"' in text
    assert 'set -- 0.0 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1' in text
    assert '-m moe_prune.code.scripts.MoDES.run_modes_ppl_grid' in text
    assert '--n-ctx 2048' in text


def test_modes_search_wrapper_names_distinguish_ppl_and_mcqa() -> None:
    ppl_alias = _read("code/scripts/MoDES/run_modes_search_pruning_rate_by_ppl.sh")
    mcqa_alias = _read("code/scripts/MoDES/run_modes_search_pruning_rate_by_mcqa.sh")
    ppl_legacy = _read("code/scripts/MoDES/run_modes_ppl_search_pruning_rate.sh")
    mcqa_legacy = _read("code/scripts/MoDES/run_modes_search_pruning_rate.sh")

    assert 'run_modes_ppl_search_pruning_rate.sh' in ppl_alias
    assert 'run_modes_search_pruning_rate.sh' in mcqa_alias
    assert 'run_modes_search_pruning_rate_by_ppl.sh' in ppl_legacy
    assert 'run_modes_search_pruning_rate_by_mcqa.sh' in mcqa_legacy


def test_modes_runtime_reuses_valid_calibration_before_recomputing() -> None:
    text = _read("code/ablation/MoDES/run_qwen3_text_pipeline.py")

    assert 'from moe_prune.code.src.modes_calibration import calibration_file_is_valid' in text
    assert 'has_valid_calibration = calibration_file_is_valid(layer_importance_path)' in text
    assert 'print(f"[MoDES-run] reusing calibration {layer_importance_path}", flush=True)' in text


def test_naee_ppl_wrapper_runs_one_beta_per_log_dir() -> None:
    text = _read("code/scripts/NAEE/run_naee_ppl_grid.sh")

    assert 'LOG_ROOT=${LOG_ROOT:-$OUTPUT_DIR/logs}' in text
    assert 'tau_dir=$(printf "tau_%.4f" "$beta")' in text
    assert "--beta-grid \"$beta\"" in text
    assert "--skip-calibration" in text
    assert 'tee -a "$log_file"' in text


def test_diep_ppl_wrapper_is_model_family_aware() -> None:
    text = _read("code/scripts/DiEP/run_diep_ppl_grid.sh")

    assert 'source "$SCRIPT_DIR/../shared/model_family.sh"' in text
    assert 'OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE_ROOT/results/DiEP/$MODEL_TAG/wikitext_search}' in text
    assert 'SCORE_CACHE_DIR=${SCORE_CACHE_DIR:-$WORKSPACE_ROOT/results/DiEP/$MODEL_TAG/calibration/score_cache}' in text


def test_diep_ppl_search_keeps_calibration_separate_from_wikitext_search() -> None:
    text = _read("code/scripts/DiEP/run_diep_ppl_search_pruning_rate.py")

    assert "args.results_root / 'calibration' / 'score_cache'" in text
    assert "calibration_dir = args.results_root / 'calibration'" in text
    assert "search_dir = args.results_root / 'wikitext_search'" in text
    assert "search_dir.mkdir(parents=True, exist_ok=True)" not in text
    assert "base_dir.mkdir(parents=True, exist_ok=True)" not in text


def test_diep_ppl_wrappers_do_not_precreate_result_dirs() -> None:
    grid_text = _read("code/scripts/DiEP/run_diep_ppl_grid.sh")
    pipeline_text = _read("code/scripts/DiEP/run_diep_qwen3_ppl_pipeline.sh")

    assert 'mkdir -p "$OUTPUT_DIR" "$SCORE_CACHE_DIR"' not in grid_text
    assert 'mkdir -p "$OUTPUT_DIR" "$SCORE_CACHE_DIR"' not in pipeline_text


def test_finalize_matrix_pipeline_covers_verification_audit_and_plots() -> None:
    text = _read("code/scripts/shared/finalize_matrix_pipeline.sh")

    assert 'run_qwen36_and_gemma4_modes_verifications' in text
    assert 'run_wikitext_remediation' in text
    assert 'audit_search_matrix' in text
    assert 'plot_search_curves' in text
    assert 'run_modes_ppl_search_pruning_rate' in text
    assert 'POLL_INTERVAL=${POLL_INTERVAL:-60}' in text
    assert 'STATE_DIR=${STATE_DIR:-"$REPO_ROOT/results/finalize_state"}' in text
    assert 'STATUS_PATH=${STATUS_PATH:-"$STATE_DIR/status.json"}' in text
    assert 'MODES_CAL_LOG=${MODES_CAL_LOG:-"$REPO_ROOT/results/logs/modes_calibration_128.log"}' in text
    assert 'AUDIT_JSON=${AUDIT_JSON:-"$REPO_ROOT/results/audit/search_matrix_audit.json"}' in text
    assert 'write_status() {' in text
    assert 'if [[ -t 1 ]]; then' in text
    assert 'printf \'[finalize-matrix] %s\\n\' "$*" >>"$LOG_DIR/finalize_matrix_pipeline.log"' in text
    assert '"pid": pid,' in text
    assert '"poll_interval_seconds": poll_interval,' in text
    assert 'payload["progress"] = progress' in text
    assert 'payload["eta_seconds"] = int(eta_seconds_raw)' in text
    assert 'payload["calibration_process"] = {' in text
    assert 'payload["incomplete_cells"] = int(audit_payload["summary"]["incomplete_cells"])' in text
    assert '"stages": {' in text
    assert 'model_path_for() {' in text
    assert 'latest_modes_calibration_progress() {' in text
    assert 'latest_modes_calibration_eta_seconds() {' in text
    assert 'modes_calibration_pid() {' in text
    assert 'archive_incomplete_dir() {' in text
    assert 'archived incomplete work_dir $work_dir -> $archive_dir' in text
    assert 'valid_sentinel() {' in text
    assert 'python - "$sentinel" <<\'PY\'' in text
    assert 'if suffix == ".json":' in text
    assert 'if suffix == ".jsonl":' in text
    assert 'python .*get_layer_importance_ddp.py --name_or_path ${model_path}' in text
    assert 'launch_modes_calibration_if_needed() {' in text
    assert 'write_status "launching_missing_calibration" "$family"' in text
    assert 'printf \'[finalize-matrix] relaunch calibration for %s on gpu=%s\\n\' "$family" "$GPU" >>"$MODES_CAL_LOG"' in text
    assert '--temperature 1.0 >>"$MODES_CAL_LOG" 2>&1 &' in text
    assert 'write_status "waiting_for_valid_calibration" "$path"' in text
    assert 'write_status "audit_poll" "results/audit/search_matrix_audit.json" "$incomplete"' in text
    assert 'modes_calibration_is_valid() {' in text
    assert 'wait_for_valid_modes_calibration "qwen3.6" "$q36_cal"' in text
    assert 'wait_for_valid_modes_calibration "gemma4" "$g4_cal"' in text
    assert 'archive_incomplete_dir "$q36_verify_sentinel" "$q36_verify_dir"' in text
    assert 'archive_incomplete_dir "$g4_verify_sentinel" "$g4_verify_dir"' in text
    assert 'if valid_sentinel "$sentinel"; then' in text
    assert 'mark_stage_done "modes_verification"' in text
    assert 'mark_stage_done "wikitext_remediation"' in text
    assert 'mark_stage_done "final_plots"' in text


def test_run_full_matrix_reuses_only_valid_modes_calibration() -> None:
    runner_text = _read("code/scripts/shared/run_full_matrix.sh")
    common_text = _read("code/scripts/shared/matrix_common.sh")
    modes_text = _read("code/scripts/MoDES/run_modes_matrix_cell.sh")

    assert 'source "$SCRIPT_DIR/matrix_common.sh"' in runner_text
    assert "modes_calibration_is_valid() {" in common_text
    assert "ensure_modes_calibration() {" in common_text
    assert 'if ! modes_calibration_is_valid "$calibration_path"; then' in modes_text
    assert 'ensure_modes_calibration "$model" "$gpu"' in modes_text
    assert 'GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-' in runner_text
    assert 'extra+=(--generation-max-tokens "$GENERATION_MAX_TOKENS")' in modes_text


def test_run_full_matrix_supports_single_gpu_serial_execution() -> None:
    text = _read("code/scripts/shared/run_full_matrix.sh")

    assert 'EXECUTION_MODE=${EXECUTION_MODE:-parallel}' in text
    assert 'SINGLE_GPU=${SINGLE_GPU:-0}' in text
    assert 'case "$EXECUTION_MODE" in' in text
    assert 'serial)' in text
    assert 'parallel)' in text
    assert 'local gpu="${4:-$(gpu_for_method "$method")}"' in text
    assert 'launch_method_queue "$method" "$SINGLE_GPU"' in text
    assert 'launch_method_queue "$method" "$(gpu_for_method "$method")" &' in text
    assert '[error] EXECUTION_MODE must be parallel or serial' in text


def test_run_full_matrix_delegates_to_method_specific_matrix_scripts() -> None:
    text = _read("code/scripts/shared/run_full_matrix.sh")

    assert 'method_script_for() {' in text
    assert '"$REPO_ROOT/code/scripts/method1/run_method1_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/method2/run_method2_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/method3/run_method3_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/method4/run_method4_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/NAEE/run_naee_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/score_only/run_score_only_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/DiEP/run_diep_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/MoDES/run_modes_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code/scripts/AIMER/run_aimer_matrix_cell.sh"' in text
    assert '"$REPO_ROOT/code_v2/scripts/run_top_p_aimer_matrix_cell.sh"' in text
    assert 'script_path=$(method_script_for "$method")' in text
    assert '"$script_path" "$model" "$dataset"' in text


def test_run_full_matrix_parallel_mode_lets_each_method_advance_independently(tmp_path: Path) -> None:
    wrapper = tmp_path / "fake_python.sh"
    log_path = tmp_path / "calls.log"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'start|%s|%s|%s\\n' \"$(date +%s.%N)\" \"${{CUDA_VISIBLE_DEVICES:-unset}}\" \"$*\" >> {log_path}\n"
        "if [[ \"$*\" == *'--method method3 '* ]]; then\n"
        "  sleep 0.6\n"
        "else\n"
        "  sleep 0.1\n"
        "fi\n"
        f"printf 'end|%s|%s|%s\\n' \"$(date +%s.%N)\" \"${{CUDA_VISIBLE_DEVICES:-unset}}\" \"$*\" >> {log_path}\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    subprocess.run(
        [
            "bash",
            "code/scripts/shared/run_full_matrix.sh",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        env={
            "PYTHON_BIN": str(wrapper),
            "MODELS": "qwen3",
            "DATASETS": "wikitext piqa",
            "METHODS": "method1 method3",
            "TARGETS": "0.1",
            "EXECUTION_MODE": "parallel",
            "PYTHONPATH": str(ROOT.parent),
        },
        capture_output=True,
    )

    records = [line.split("|", 3) for line in log_path.read_text(encoding="utf-8").splitlines()]
    method1_wikitext_end = next(
        index for index, record in enumerate(records) if record[0] == "end" and "--method method1 " in record[3] and "--dataset wikitext " in record[3]
    )
    method3_wikitext_end = next(
        index for index, record in enumerate(records) if record[0] == "end" and "--method method3 " in record[3] and "--dataset wikitext " in record[3]
    )
    method1_piqa_start = next(
        index for index, record in enumerate(records) if record[0] == "start" and "--method method1 " in record[3] and "--dataset piqa " in record[3]
    )

    assert method1_piqa_start < method3_wikitext_end
    assert method1_wikitext_end < method3_wikitext_end


def test_method_specific_matrix_scripts_exist_for_all_supported_methods() -> None:
    expected_paths = [
        ROOT / "code/scripts/method1/run_method1_matrix_cell.sh",
        ROOT / "code/scripts/method2/run_method2_matrix_cell.sh",
        ROOT / "code/scripts/method3/run_method3_matrix_cell.sh",
        ROOT / "code/scripts/method4/run_method4_matrix_cell.sh",
        ROOT / "code/scripts/NAEE/run_naee_matrix_cell.sh",
        ROOT / "code/scripts/score_only/run_score_only_matrix_cell.sh",
        ROOT / "code/scripts/DiEP/run_diep_matrix_cell.sh",
        ROOT / "code/scripts/MoDES/run_modes_matrix_cell.sh",
        ROOT / "code/scripts/AIMER/run_aimer_matrix_cell.sh",
        ROOT / "code_v2/scripts/run_top_p_aimer_matrix_cell.sh",
    ]

    for path in expected_paths:
        assert path.exists(), f"missing matrix cell script: {path}"


def test_method_specific_matrix_scripts_keep_method_logic_local() -> None:
    method1_text = _read("code/scripts/method1/run_method1_matrix_cell.sh")
    method4_text = _read("code/scripts/method4/run_method4_matrix_cell.sh")
    naee_text = _read("code/scripts/NAEE/run_naee_matrix_cell.sh")
    score_only_text = _read("code/scripts/score_only/run_score_only_matrix_cell.sh")
    diep_text = _read("code/scripts/DiEP/run_diep_matrix_cell.sh")
    modes_text = _read("code/scripts/MoDES/run_modes_matrix_cell.sh")
    aimer_text = _read("code/scripts/AIMER/run_aimer_matrix_cell.sh")
    top_p_aimer_text = _read("code_v2/scripts/run_top_p_aimer_matrix_cell.sh")

    assert '--method "method1"' in method1_text
    assert '-m moe_prune.code.scripts.shared.run_ppl_search' in method1_text
    assert '-m moe_prune.code.scripts.shared.run_dataset_search' in method1_text
    assert '--search-mode quantile' in method1_text
    assert '--method "method4"' in method4_text
    assert '-m moe_prune.code.scripts.shared.run_ppl_search' in method4_text
    assert '-m moe_prune.code.scripts.shared.run_dataset_search' in method4_text
    assert '--search-mode quantile' in method4_text
    assert '--method "naee"' in naee_text
    assert '--dataset "$dataset"' in naee_text
    assert '--search-mode quantile' in naee_text
    assert '--method "score_only"' in score_only_text
    assert '--search-mode quantile' in score_only_text
    assert 'ensure_diep_score "$model" "$gpu"' in diep_text
    assert '--score-path "$score_path"' in diep_text
    assert 'ensure_modes_calibration "$model" "$gpu"' in modes_text
    assert '--layer-importance-path "$calibration_path"' in modes_text
    assert '--method "aimer"' in aimer_text
    assert '-m moe_prune.code.scripts.shared.run_ppl_search' in aimer_text
    assert '-m moe_prune.code.scripts.shared.run_dataset_search' in aimer_text
    assert '--search-mode quantile' in aimer_text
    assert '--method "top_p_aimer"' in top_p_aimer_text
    assert '-m moe_prune.code.scripts.shared.run_ppl_search' in top_p_aimer_text
    assert '-m moe_prune.code.scripts.shared.run_dataset_search' in top_p_aimer_text
    assert '--search-mode quantile' in top_p_aimer_text


def test_run_full_matrix_serial_mode_does_not_overlap_methods(tmp_path: Path) -> None:
    wrapper = tmp_path / "fake_python.sh"
    log_path = tmp_path / "calls.log"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'start|%s|%s|%s\\n' \"$(date +%s.%N)\" \"${{CUDA_VISIBLE_DEVICES:-unset}}\" \"$*\" >> {log_path}\n"
        "sleep 0.2\n"
        f"printf 'end|%s|%s|%s\\n' \"$(date +%s.%N)\" \"${{CUDA_VISIBLE_DEVICES:-unset}}\" \"$*\" >> {log_path}\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    subprocess.run(
        [
            "bash",
            "code/scripts/shared/run_full_matrix.sh",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        env={
            "PYTHON_BIN": str(wrapper),
            "MODELS": "qwen3",
            "DATASETS": "wikitext",
            "METHODS": "method1 method3 score_only",
            "TARGETS": "0.1",
            "EXECUTION_MODE": "serial",
            "SINGLE_GPU": "7",
            "PYTHONPATH": str(ROOT.parent),
        },
        capture_output=True,
    )

    records = [line.split("|", 3) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record[0] for record in records] == ["start", "end", "start", "end", "start", "end"]
    assert [record[2] for record in records] == ["7", "7", "7", "7", "7", "7"]
    commands = [record[3] for record in records[::2]]
    assert "--method method1 " in commands[0]
    assert "--method method3 " in commands[1]
    assert "--method score_only " in commands[2]


def test_evalscope_search_entrypoints_archive_incomplete_workdirs_before_rerun() -> None:
    dataset_text = _read("code/scripts/shared/run_dataset_search.py")
    search_text = _read("code/scripts/shared/run_evalscope_search.py")
    worker_text = _read("code/scripts/shared/run_evalscope_search_worker.sh")

    assert 'archive_incomplete_work_dir' in dataset_text
    assert "archived incomplete work_dir" in dataset_text
    assert 'archive_incomplete_work_dir' in search_text
    assert "archived incomplete work_dir" in search_text
    assert 'SEARCH_MODE=quantile' in worker_text
    assert 'SEARCH_MODE_INPUT=${SEARCH_MODE:-auto}' in worker_text
    assert 'default_search_mode_for_method' in worker_text


def test_ppl_search_entrypoints_archive_incomplete_workdirs_before_rerun() -> None:
    shared_text = _read("code/scripts/shared/run_ppl_search.py")
    modes_text = _read("code/scripts/MoDES/run_modes_ppl_search_pruning_rate.py")
    diep_text = _read("code/scripts/DiEP/run_diep_ppl_search_pruning_rate.py")
    eat_text = _read("code/scripts/EAT-MOE/run_eat_moe_ppl_search_pruning_rate.py")
    worker_text = _read("code/scripts/shared/run_ppl_search_worker.sh")

    assert 'archive_incomplete_work_dir' in shared_text
    assert "archived incomplete work_dir" in shared_text
    assert 'archive_incomplete_work_dir' in modes_text
    assert "archived incomplete work_dir" in modes_text
    assert 'archive_incomplete_work_dir' in diep_text
    assert "archived incomplete work_dir" in diep_text
    assert 'archive_incomplete_work_dir' in eat_text
    assert "archived incomplete work_dir" in eat_text
    assert 'SEARCH_MODE=quantile' in worker_text
    assert 'SEARCH_MODE_INPUT=${SEARCH_MODE:-auto}' in worker_text
    assert 'default_search_mode_for_method' in worker_text


def test_specialized_eval_search_entrypoints_archive_incomplete_workdirs_before_rerun() -> None:
    eat_text = _read("code/scripts/EAT-MOE/run_eat_moe_search_pruning_rate.py")
    modes_text = _read("code/scripts/MoDES/run_modes_search_pruning_rate.py")
    diep_text = _read("code/scripts/DiEP/run_diep_search_pruning_rate.py")

    assert 'archive_incomplete_work_dir' in eat_text
    assert "archived incomplete work_dir" in eat_text
    assert 'archive_incomplete_work_dir' in modes_text
    assert "archived incomplete work_dir" in modes_text
    assert 'archive_incomplete_work_dir' in diep_text
    assert "archived incomplete work_dir" in diep_text

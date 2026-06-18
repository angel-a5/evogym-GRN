#!/usr/bin/env python3
import argparse
import json
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from algorithms.EA_classes import Individual
from algorithms.GRN_2D import GRN
from simulation.prepare_robot_files import prepare_robot_files
from simulation.simulation_resources import simulate_evogym_batch
PARAM_KEYS = [
    "out_path",
    "study_name",
    "experiments",
    "runs",
    "voxel_types",
    "env_conditions",
    "plastic",
    "max_voxels",
    "cube_face_size",
    "evogym_steps",
    "evogym_init_x",
    "evogym_init_y",
    "evogym_action_bias",
    "evogym_action_amplitude",
    "evogym_period_steps",
    "evogym_render_mode",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize top robots from experiment DBs in EvoGym."
    )
    p.add_argument(
        "--params_file",
        default=str(ROOT / "experiments" / "locomotion.sh"),
        help="Path to experiment params .sh file.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of best robots to visualize globally.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="fitness",
        help="Ranking metric (deprecated here; visualization selection uses fitness).",
    )
    p.add_argument(
        "--ascending",
        type=int,
        default=0,
        help="Set to 1 to select lowest values first.",
    )
    p.add_argument(
        "--rank_mode",
        type=str,
        choices=["best", "worst"],
        default="best",
        help="Ranking direction for fitness selection.",
    )
    p.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Optional override for EvoGym render mode.",
    )
    p.add_argument(
        "--generations",
        type=str,
        default="100",
        help="Optional CSV generation filter, e.g. '10,20,50'. Empty means best-ever.",
    )
    return p.parse_args()


def _split_csv(text):
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _load_params_from_shell(params_path: Path):
    if not params_path.exists():
        raise FileNotFoundError(f"params file not found: {params_path}")

    keys = " ".join(PARAM_KEYS)
    cmd = (
        f"set -a; source {shlex.quote(str(params_path))}; "
        f"for k in {keys}; do printf '%s=%s\\n' \"$k\" \"${{!k-}}\"; done"
    )
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        text=True,
        capture_output=True,
    )

    out = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v
    return out


def _pick_for_experiment(values, exp_idx, field_name):
    if not values:
        raise ValueError(f"Missing required field: {field_name}")
    if len(values) == 1:
        return values[0]
    if exp_idx >= len(values):
        raise ValueError(
            f"Field '{field_name}' must have either one value or one per experiment."
        )
    return values[exp_idx]


def _db_path(out_path, study_name, experiment_name, run):
    ###################################################################################
    return Path(out_path) / study_name / experiment_name / f"run_{run}" / f"run_{run}"
    ###################################################################################


def _table_columns(cur, table_name):
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def _fetch_top_from_db(db_path: Path, metric: str, top_k: int, ascending: bool, generations=None):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    robot_cols = _table_columns(cur, "all_robots")
    surv_cols = _table_columns(cur, "generation_survivors")
    direction = "ASC" if ascending else "DESC"
    gen_filter = [int(g) for g in (generations or [])]
    gen_sql = ""
    gen_params = []
    if gen_filter:
        placeholders = ",".join(["?"] * len(gen_filter))
        gen_params = gen_filter

    # If generations are provided, treat generation_survivors as source of truth
    # for membership in those generations (join to all_robots for genome/abs metrics).
    if gen_filter:
        if metric in surv_cols:
            metric_expr = f"gs.{metric}"
        elif metric in robot_cols:
            metric_expr = f"r.{metric}"
        else:
            conn.close()
            valid = sorted(set(robot_cols) | set(surv_cols))
            raise ValueError(
                f"Metric '{metric}' not found in DB columns. Available examples: {valid[:20]}"
            )

        rows = cur.execute(
            f"""
            SELECT
                r.robot_id AS robot_id,
                r.genome AS genome,
                {metric_expr} AS metric_value,
                gs.generation AS generation
            FROM generation_survivors gs
            JOIN all_robots r ON r.robot_id = gs.robot_id
            WHERE gs.generation IN ({placeholders})
              AND {metric_expr} IS NOT NULL
            ORDER BY gs.generation ASC, metric_value {direction}
            """,
            tuple(gen_params),
        ).fetchall()
    elif metric in robot_cols:
        rows = cur.execute(
            f"""
            SELECT robot_id, genome, {metric} AS metric_value, born_generation AS generation
            FROM all_robots
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} {direction}
            LIMIT ?
            """,
            (top_k,),
        ).fetchall()
    elif metric in surv_cols:
        rows = cur.execute(
            f"""
            SELECT
                r.robot_id AS robot_id,
                r.genome AS genome,
                gs.{metric} AS metric_value,
                gs.generation AS generation
            FROM all_robots r
            JOIN generation_survivors gs ON gs.robot_id = r.robot_id
            WHERE gs.{metric} IS NOT NULL
            ORDER BY metric_value {direction}
            LIMIT ?
            """,
            (top_k,),
        ).fetchall()
    else:
        conn.close()
        valid = sorted(set(robot_cols) | set(surv_cols))
        raise ValueError(
            f"Metric '{metric}' not found in DB columns. Available examples: {valid[:20]}"
        )

    conn.close()
    out = []
    for row in rows:
        genome = row["genome"]
        if isinstance(genome, str):
            genome = json.loads(genome)
        out.append(
            {
                "robot_id": int(row["robot_id"]),
                "genome": genome,
                "metric_value": float(row["metric_value"]),
                "generation": row["generation"],
            }
        )
    return out


def _build_phenotype(genome, max_voxels, cube_face_size, voxel_types, env_conditions, plastic, use_diffusion): #diffusion added
    cells = GRN(
        promoter_threshold=0.95,
        max_voxels=max_voxels,
        cube_face_size=cube_face_size,
        genotype=genome,
        voxel_types=voxel_types,
        env_conditions=env_conditions,
        plastic=plastic,
        ############################
        use_diffusion=use_diffusion,
        ############################
    ).develop()
    materials = np.zeros(cells.shape, dtype=int)
    for idx, value in np.ndenumerate(cells):
        materials[idx] = value.voxel_type if value != 0 else 0
    return materials


def main():
    args = parse_args()
    params = _load_params_from_shell(Path(args.params_file).resolve())

    experiments = _split_csv(params.get("experiments"))
    runs = _split_csv(params.get("runs"))
    voxel_types_list = _split_csv(params.get("voxel_types"))
    env_conditions_list = _split_csv(params.get("env_conditions"))

    if not experiments:
        raise ValueError("No experiments found in params file.")
    if not runs:
        raise ValueError("No runs found in params file.")

    top_k = max(1, int(args.top_k))
    if str(args.metric).strip().lower() != "fitness":
        print("[info] --metric is ignored in this viewer; using fitness ranking.")
    metric = "fitness"
    if int(args.ascending) != 0:
        print("[info] --ascending is deprecated in this viewer; use --rank_mode best|worst.")
    ascending = args.rank_mode == "worst"
    gen_filter = [int(x) for x in _split_csv(args.generations)]

    selected = []
    for exp_idx, exp_name in enumerate(experiments):
        voxel_types = _pick_for_experiment(voxel_types_list, exp_idx, "voxel_types")
        env_conditions = _pick_for_experiment(env_conditions_list, exp_idx, "env_conditions")

        for run_order, run in enumerate(runs):
            db_path = _db_path(params["out_path"], params["study_name"], exp_name, run)
            print("RUN =", run)
            print("DB PATH =", db_path)
            if not db_path.exists():
                print(f"[skip] DB missing: {db_path}")
                continue

            rows = _fetch_top_from_db(
                db_path,
                metric=metric,
                top_k=top_k,
                ascending=ascending,
                generations=gen_filter,
            )
            if not rows:
                continue

            if gen_filter:
                # top_k per requested generation, in generation order.
                for g in sorted(set(gen_filter)):
                    picked = 0
                    for row in rows:
                        if row["generation"] != g:
                            continue
                        row["experiment_name"] = exp_name
                        row["run"] = run
                        row["voxel_types"] = voxel_types
                        row["env_conditions"] = env_conditions
                        row["exp_order"] = exp_idx
                        row["run_order"] = run_order
                        selected.append(row)
                        picked += 1
                        if picked >= top_k:
                            break
                    if picked == 0:
                        print(f"[warn] No survivors for exp={exp_name} run={run} generation={g}")
            else:
                # top_k over all generations for each run.
                for row in rows[:top_k]:
                    row["experiment_name"] = exp_name
                    row["run"] = run
                    row["voxel_types"] = voxel_types
                    row["env_conditions"] = env_conditions
                    row["exp_order"] = exp_idx
                    row["run_order"] = run_order
                    selected.append(row)

    if not selected:
        print("No candidates found.")
        return

    # Deterministic playback order:
    # 1) experiment order from params
    # 2) run order from params
    # 3) if generations are requested, generation ascending
    # 4) fitness rank in requested direction
    if gen_filter:
        if ascending:
            selected.sort(
                key=lambda x: (
                    x["exp_order"],
                    x["run_order"],
                    x["generation"] if x["generation"] is not None else 10**9,
                    x["metric_value"],
                )
            )
        else:
            selected.sort(
                key=lambda x: (
                    x["exp_order"],
                    x["run_order"],
                    x["generation"] if x["generation"] is not None else 10**9,
                    -x["metric_value"],
                )
            )
    else:
        if ascending:
            selected.sort(key=lambda x: (x["exp_order"], x["run_order"], x["metric_value"]))
        else:
            selected.sort(key=lambda x: (x["exp_order"], x["run_order"], -x["metric_value"]))

    # Visualization config (always with graphics, one robot at a time).
    vis_args = SimpleNamespace(
        voxel_types="withbone",
        out_path=params.get("out_path", "tmp_out"),
        study_name=params.get("study_name", "defaultstudy"),
        experiment_name="viz_replay",
        run=0,
        evogym_steps=int(params.get("evogym_steps") or 500),
        evogym_num_workers=1,
        evogym_init_x=int(params.get("evogym_init_x") or 3),
        evogym_init_y=int(params.get("evogym_init_y") or 1),
        evogym_action_bias=float(params.get("evogym_action_bias") or 1.0),
        evogym_action_amplitude=float(params.get("evogym_action_amplitude") or 0.4),
        evogym_period_steps=int(params.get("evogym_period_steps") or 20),
        evogym_headless=0,
        evogym_render_mode=args.render_mode or params.get("evogym_render_mode") or "screen",
    )

    print(f"Params file: {args.params_file}")
    print(f"Study: {params.get('study_name')}")
    print(f"Experiments: {experiments}")
    print(f"Runs: {runs}")
    if gen_filter:
        print(f"Generations filter: {gen_filter}")
    else:
        print("Generations filter: best-ever (all generations)")
    print(f"Ranking metric: {metric} ({args.rank_mode})")
    if gen_filter:
        print("Playback order: experiment -> run -> generation -> fitness")
        print(f"top_k per generation: {top_k}")
    else:
        print("Playback order: experiment -> run -> fitness")
        print(f"top_k per run (all generations): {top_k}")
    print(f"Selected robots total: {len(selected)}")

    for rank, entry in enumerate(selected, start=1):
        exp_name = entry["experiment_name"]
        run = entry["run"]

        ######################################################
        use_diffusion = "noDiff" not in str(run)
        print(f"Replay diffusion enabled: {use_diffusion}")
        ######################################################

        rid = entry["robot_id"]
        score = entry["metric_value"]
        generation = entry["generation"]
        voxel_types = entry["voxel_types"]
        env_conditions = entry["env_conditions"]
        plastic = int(params.get("plastic") or 0)
        max_voxels = int(params.get("max_voxels") or 64)
        cube_face_size = int(params.get("cube_face_size") or 4)

        print(
            f"\n[{rank}/{len(selected)}] exp={exp_name} run={run} "
            f"robot_id={rid} fitness_from_db={score:.6f} generation={generation}"
        )

        ind = Individual(genome=entry["genome"], id_counter=rid)
        ind.valid = 1
        ind.phenotype = _build_phenotype(
            genome=entry["genome"],
            max_voxels=max_voxels,
            cube_face_size=cube_face_size,
            voxel_types=voxel_types,
            env_conditions=env_conditions,
            plastic=plastic,
            #############################
            use_diffusion=use_diffusion,
            #############################
        )

        vis_args.voxel_types = voxel_types
        prepare_robot_files(ind, vis_args)
        simulate_evogym_batch([ind], vis_args)
        print(f"Replay displacement={ind.displacement:.6f}")


if __name__ == "__main__":
    main()

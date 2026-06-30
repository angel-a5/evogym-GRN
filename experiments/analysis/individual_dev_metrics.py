# I needed to make this since I forgot to add the individual-level metrics before I did my 20 runs

import csv
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.GRN_2D import GRN

MAX_VOXELS = 125
CUBE_FACE_SIZE = 5
VOXEL_TYPES = "withbone"
ENV_CONDITIONS = "none"
PLASTIC = 0
PROMOTER_THRESHOLD = 0.95

RUNS_DIR = ROOT / "tmp_out" / "evobots" / "evobots"
OUT_DIR = ROOT.parent / "metrics"

ACTIVE_TF_THRESHOLD = 1.0


def active_tfs(tf_state, threshold=ACTIVE_TF_THRESHOLD):
    return {tf for tf, value in tf_state.items() if value >= threshold}


def trajectory_variability(history):
    if not history or len(history) < 2:
        return 0.0
    total = 0.0
    for t in range(len(history) - 1):
        state_a = history[t]
        state_b = history[t + 1]
        all_tfs = set(state_a.keys()) | set(state_b.keys())
        for tf in all_tfs:
            total += abs(state_b.get(tf, 0.0) - state_a.get(tf, 0.0))
    return total


def develop_one(args):
    robot_id, born_generation, parent1_id, parent2_id, genome_json, use_diffusion = args
    genome = json.loads(genome_json) if isinstance(genome_json, str) else genome_json

    grn = GRN(
        promoter_threshold=PROMOTER_THRESHOLD,
        max_voxels=MAX_VOXELS,
        cube_face_size=CUBE_FACE_SIZE,
        genotype=genome,
        voxel_types=VOXEL_TYPES,
        env_conditions=ENV_CONDITIONS,
        plastic=PLASTIC,
        use_diffusion=use_diffusion,
    )
    grn.develop()

    tf_final = grn.tf_history[-1] if grn.tf_history else {}
    tf_concentration = sum(tf_final.values())
    num_active_tfs = len(active_tfs(tf_final))
    traj_var = trajectory_variability(grn.tf_history)

    return (
        robot_id,
        born_generation,
        parent1_id,
        parent2_id,
        round(tf_concentration, 4),
        num_active_tfs,
        round(traj_var, 4),
    )


def process_run(run_dir_name):
    run_dir = RUNS_DIR / run_dir_name
    db_path = run_dir / run_dir_name
    if not db_path.exists():
        print(f"[skip] no db at {db_path}")
        return

    use_diffusion = "noDiff" not in run_dir_name

    parts = run_dir_name.replace("run_run_", "").split("_")
    n = parts[0]
    tag = "diff" if use_diffusion else "noDiff"
    out_name = f"individual_developmental_metrics_{tag}{n}.csv"
    out_path = OUT_DIR / out_name

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT seed FROM experiment_info LIMIT 1")
    seed_row = cur.fetchone()
    seed = seed_row[0] if seed_row else None

    cur.execute(
        "SELECT robot_id, born_generation, parent1_id, parent2_id, genome FROM all_robots"
    )
    rows = cur.fetchall()
    con.close()

    tasks = [
        (robot_id, born_generation, parent1_id, parent2_id, genome, use_diffusion)
        for robot_id, born_generation, parent1_id, parent2_id, genome in rows
    ]

    print(f"[{run_dir_name}] developing {len(tasks)} individuals...")
    t0 = time.time()
    with Pool() as pool:
        results = pool.map(develop_one, tasks, chunksize=50)
    print(f"[{run_dir_name}] done in {time.time() - t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "generation",
                "individual_id",
                "parent1_id",
                "parent2_id",
                "tf_concentration",
                "num_active_tfs",
                "trajectory_variability",
            ]
        )
        for robot_id, generation, p1, p2, tf_conc, n_active, traj in results:
            writer.writerow([seed, generation, robot_id, p1, p2, tf_conc, n_active, traj])

    print(f"[{run_dir_name}] wrote {out_path}")


def main():
    run_dirs = sorted(
        d.name
        for d in RUNS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_run_") and "_test" not in d.name
    )
    print(f"Found {len(run_dirs)} run directories: {run_dirs}")
    for run_dir_name in run_dirs:
        process_run(run_dir_name)


if __name__ == "__main__":
    main()

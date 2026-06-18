# experiment.py
import os, sys
import random
import sqlite3
from sqlalchemy import create_engine, func, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.EA_classes import Base, Robot, GenerationSurvivor, Individual, ExperimentInfo
from utils.metrics import METRICS_ABS, METRICS_REL


# Enable FK enforcement in SQLite (otherwise FK errors won't trip the transaction)
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


class Experiment:
    """
    Handles experiment bookkeeping:
      - output/db paths
      - SQLAlchemy engine/session
      - RNG seed management
      - state recovery from DB
      - atomic persistence per generation
    """

    def __init__(self, args):
        # paths
        self.out_path = f"{args.out_path}/{args.study_name}/{args.experiment_name}/run_{args.run}"
        os.makedirs(self.out_path, exist_ok=True)
        self.db_path = os.path.join(self.out_path, f'run_{args.run}')
        self.voxel_types = args.voxel_types

    def recover_db(self):
        # by default sqlalquemy does not overwrite db, but recovers it if existent instead
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session = self.Session()

        # RNG (seed is persisted in the DB for reproducibility)
        self.rng = random.Random()
        info = self.session.query(ExperimentInfo).first()
        if info is None:
            #seed = random.randint(0, 2**32 - 1)
            seed = 1675610640
            print("seed (new)", seed)
            self.seed = seed
            self.rng.seed(seed)
            self.session.add(ExperimentInfo(seed=seed))
            self.session.commit()
        else:
            print("seed (reused)", info.seed)
            self.seed = info.seed
            self.rng.seed(info.seed)

        # running ID counter for Individuals/Robots
        self.id_counter = 0

    # ---------- Recovery ----------

    def _individual_from_robot(self, r: Robot) -> Individual:
        ind = Individual(genome=r.genome, id_counter=r.robot_id,
                         parent1_id=r.parent1_id, parent2_id=r.parent2_id)
        ind.valid = r.valid
        ind.born_generation = r.born_generation

        # copy absolute metrics exactly as stored
        for m in METRICS_ABS:
            setattr(ind, m, getattr(r, m, None))

        # copy relative metrics exactly as stored
        for m in METRICS_REL:
            setattr(ind, m, getattr(r, m, None))

        return ind

    def _recover_state(self):
        """
        Returns (last_completed_generation, recovered_population or None).

        If there is no completed generation, returns (None, None).
        Requires subclass to implement `develop_phenotype(genome)`.
        """
        with self.Session() as s:
            last_gen = s.query(func.max(GenerationSurvivor.generation)).scalar()
            if last_gen is None:
                # Assert the invariant: no robots should exist either
                if s.query(Robot).count() != 0:
                    raise RuntimeError(
                        "DB inconsistent: robots exist but no survivors. Clean or migrate."
                    )
                self.id_counter = 0
                return None, None

            # Rebuild population = survivors from last completed generation
            rows = (
                s.query(Robot, GenerationSurvivor)
                .join(GenerationSurvivor, GenerationSurvivor.robot_id == Robot.robot_id)
                .filter(GenerationSurvivor.generation == last_gen)
                .all()
            )

            population = []
            for r, gs in rows:
                ind = self._individual_from_robot(r)
                # rebuild phenotype
                ind.phenotype = self.develop_phenotype(ind, self.voxel_types)

                # --- pass-through for relative metrics ---
                for m in METRICS_REL:
                    setattr(ind, m, getattr(gs, m, None))
                population.append(ind)

            # Set next ID
            max_id = s.query(func.max(Robot.robot_id)).scalar()
            self.id_counter = int(max_id) if max_id is not None else 0

            return int(last_gen), population

    # ---------- Persistence (one atomic save per generation) ----------

    def _persist_generation_atomic(self, generation, robots_this_gen, survivors_this_gen):
        # Use a fresh session per generation to keep transactions clean
        with self.Session() as s, s.begin():  # s.begin() = single atomic transaction
            # Stage robot rows first (so FK to robot_id exists when survivors insert)
            for ind in robots_this_gen:
                self._stage_robot(s, ind)
            s.flush()  # optional: surfaces issues before adding survivors

            # Stage survivors
            self._stage_generation_survivors(s, generation, survivors_this_gen)
            # exiting the with-block commits; any exception rolls back everything

    def _stage_robot(self, s, individual):
        row = s.get(Robot, individual.id)
        if row is None:
            data = {
                "robot_id": individual.id,
                "born_generation": int(individual.born_generation),
                "genome": individual.genome,
                "valid": individual.valid,
                "parent1_id": individual.parent1_id,
                "parent2_id": individual.parent2_id,
            }
            # absolute metrics
            for m in METRICS_ABS:
                data[m] = getattr(individual, m, None)
            s.add(Robot(**data))

    def _stage_generation_survivors(self, s, generation, survivors):
        for ind in survivors:
            data = {
                "generation": int(generation),
                "robot_id": int(ind.id),
            }
            # relative metrics
            for m in METRICS_REL:
                data[m] = getattr(ind, m, None)
            s.merge(GenerationSurvivor(**data))


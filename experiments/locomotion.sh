#!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
out_path="tmp_out"
# /home/ripper8/projects/working_data

# DO NOT use underline ( _ ) in the study and experiments names
# delimiter of three vars below is coma. example:
#experiments="exp1,epx2"
# exps order is the same for all three vars
# exps names should not be fully contained in each other

study_name="evobots"
experiments="evobots"

# one voxel_types definition per experiment
voxel_types="withbone"

# one set of conditions per experiment
env_conditions="none"

ustatic="1"
udynamic="0.8"

####

nruns=1

# runs=""
# for i in $(seq 1 $nruns);
# do
#   runs="${runs}${i},"
# done
# runs="${runs%,}"

# Change the name of the run here, say diff in the name for diffusion, noDiff for no diffusion
runs="run_10_diff"

watchruns=$runs

algorithm="basic_EA"

fitness_metric="displacement"

plastic=0

num_generations="100"

population_size="100"

offspring_size="100"

RUN_ANALYSIS=0

# gens for box-plots, snapshots, videos (by default the last gen)
#generations="1,$num_generations"
generations="1,$num_generations"

# max gen to filter line-plots  (by default the last gen)
final_gen="$num_generations"

mutation_prob=0.9

crossover_prob=1

max_voxels=125

cube_face_size=5

evogym_steps=500

# Single parallelism knob for simulation:
# 0 = auto (uses available CPU cores), N = fixed worker count.
evogym_num_workers=0

run_simulation=1

### PARAMS END ###

import subprocess


env = """
eval "$(/home/wilmot/.software/miniconda/miniconda3/bin/conda shell.bash hook)" ;
export COPPELIASIM_ROOT=/home/aecgroup/aecdata/Software/CoppeliaSim_Edu_V4_2_0_Ubuntu16_04/ ;
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT ;
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT ;
export COPPELIASIM_MODEL_PATH=/home/wilmot/Documents/code/coppelia_sim_inverse_model/3d_models/ ;
"""


def run_experiment(delay, **kwargs):
    stdin = env + " \\\n\t".join([
        "python cluster.py --multirun",
        # "procedure.database=thesis_experiments.db",
        # "experiment.repetition=0,1,2",
    ] + [f"{key}={val}" for key, val in kwargs.items()])
    if delay == "now":
        at_cmd = 'sh'
    else:
        # at_cmd = f"at midnight + {delay} days"
        at_cmd = f"at 6pm + {delay} days"
    print(stdin + '  |  ' + at_cmd)
    cp = subprocess.run(at_cmd, input=stdin, encoding='ascii', capture_output=True, shell=True)
    print(cp.stdout)
    print(cp.stderr)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
experiments = [
    ############################################################################
    # -- -- plain old TD3
    ############################################################################
    { # standard_td3_vary_exploration_std
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"standard_vary_exploration_std",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"1.0",
        "agent.movement_exploration_stddev":"0.05,0.1,0.2,0.3,0.4,0.5"
    },

    { # standard_td3_vary_exploration_prob with exploration std = 0.5
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"standard_vary_exploration_prob",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",  ### ,1.0 removed because already present in collection standard_vary_exploration_std
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85",
    },

    ############################################################################
    # -- -- TD3 + bell shaped curves
    ############################################################################
    { # missing vary exploration std
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"bell_vary_exploration_std",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"minimalist",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"1.0",
        "agent.movement_exploration_stddev":"0.05,0.1,0.2,0.3,0.4,0.5",
    },

    { # missing vary exploration prob with exploration std = 0.5
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"bell_vary_exploration_prob",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"minimalist",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85",  ### ,1.0 removed because already present in collection bell_vary_exploration_std
    },


    ############################################################################
    # -- -- TD3 + action sequence (without bottleneck)
    ############################################################################
    { # missing vary sequence length with exploration std = 0.5 / exploration prob = 1.0
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_vary_sequence_length",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2,0.4,0.6,1.0,1.2,2.0,3.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"1.0",
        "agent.movement_exploration_stddev":"0.5",
    },

    { # missing vary exploration std with sequence_length = 1.0 exploration_prob = 1.0
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_vary_exploration_std",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"1.0",
        "agent.movement_exploration_stddev":"0.05,0.1,0.2,0.3,0.4",  ### ,0.5 removed because already present in collection trajectory_vary_sequence_length
    },

    { # missing vary exploration prob with exploration_std = 0.5
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_vary_exploration_prob",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85,1.0",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck)
    ############################################################################
    { # missing vary bottleneck size
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_bn_vary_bn",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"25,15,10,7,5,3",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"1.0",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck) + H critic
    ############################################################################
    { # missing vary exploration ratio with bottleneck_size = 5
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "procedure.primitive_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"hrl_vary_exploration_ratio",
        "agent/policy_model_arch":"movement_primitive_3_3",
        "agent.policy_bottleneck_size":"5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"1.0",
        "agent.movement_exploration_prob_ratio":"0.0,0.1,0.2,0.5,0.8,0.9,1.0",
        "agent.movement_exploration_stddev":"0.5",
        "agent.primitive_exploration_stddev":"0.5",
    },

    { # missing vary exploration prob with explotration_ratio = ???
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0",
        "procedure.primitive_noise_magnitude_limit":"0",
        "agent.tau":"0.01",
        "procedure.collection":"hrl_vary_exploration_prob",
        "agent/policy_model_arch":"movement_primitive_3_3",
        "agent.policy_bottleneck_size":"5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_prob_ratio":"0.5",
        "agent.movement_exploration_stddev":"0.5",
        "agent.primitive_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85",  ### ,1.0 removed because already present in collection trajectory_vary_sequence_length
    },
]


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
experiments_with_ada_lambda_step = [
    ############################################################################
    # -- -- with Ada Lambda Step -- --
    ############################################################################
    { # standard_td3_vary_exploration_prob with exploration std = 0.5 and with Ada Lambda Step (=0.01)
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"standard_vary_exploration_prob_ada_small",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85,1.0",
    },

    ############################################################################
    # -- -- TD3 + bell shaped curves
    ############################################################################
    { # missing vary exploration prob with exploration std = 0.5
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"bell_vary_exploration_prob_ada_small",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"minimalist",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85,1.0",
    },


    ############################################################################
    # -- -- TD3 + action sequence (without bottleneck)
    ############################################################################
    { # missing vary exploration prob with exploration_std = 0.5
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_vary_exploration_prob_ada_small",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85,1.0",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck)
    ############################################################################
    { # missing vary bottleneck size
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_bn_vary_bn_ada_small",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"25,15,10,7,5,3",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.4",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck) + H critic
    ############################################################################
    { # missing vary exploration ratio with bottleneck_size = 5
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "procedure.primitive_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"hrl_vary_exploration_ratio_ada_small",
        "agent/policy_model_arch":"movement_primitive_3_3",
        "agent.policy_bottleneck_size":"5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"1.0",
        "agent.movement_exploration_prob_ratio":"0.0,0.1,0.2,0.5,0.8,0.9,1.0",
        "agent.movement_exploration_stddev":"0.5",
        "agent.primitive_exploration_stddev":"0.5",
    },

    { # missing vary exploration prob with explotration_ratio = ???
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "procedure.primitive_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"hrl_vary_exploration_prob_ada_small",
        "agent/policy_model_arch":"movement_primitive_3_3",
        "agent.policy_bottleneck_size":"5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_prob_ratio":"0.5",
        "agent.movement_exploration_stddev":"0.5",
        "agent.primitive_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1,0.25,0.4,0.55,0.7,0.85",  ### ,1.0 removed because already present in collection trajectory_vary_sequence_length
    },
]

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


experiments = [
    ############################################################################
    # -- -- plain old TD3
    ############################################################################
    { # standard_td3_vary_exploration_prob with exploration std = 0.5
        "procedure.database":"thesis_experiments_v2.db",
        "experiment.repetition":"0,1,2,3",
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.5",
        # "procedure.movement_noise_magnitude_limit":"0.01",
        # "procedure.movement_noise_magnitude_limit":"0,0.01,0.5",
        "agent.tau":"0.01",
        "procedure.collection":"standard",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",
        # "agent.exploration_prob":"1.0",
        "agent.exploration_prob":"0.1",
        # "agent.exploration_prob":"0.1,1.0",
    },

    ############################################################################
    # -- -- TD3 + bell shaped curves
    ############################################################################
    { # missing vary exploration std
        "procedure.database":"thesis_experiments_v2.db",
        "experiment.repetition":"0,1,2",
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0,0.01,0.5",
        "agent.tau":"0.01",
        "procedure.collection":"bell",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"minimalist",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1,1.0",
        "agent.movement_exploration_stddev":"0.2",
    },

    ############################################################################
    # -- -- TD3 + action sequence (without bottleneck)
    ############################################################################
    { # missing vary sequence length with exploration std = 0.5 / exploration prob = 1.0
        "procedure.database":"thesis_experiments_v2.db",
        "experiment.repetition":"0,1,2,3",
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0,0.01,0.5",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.6,1.0,1.2",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck)
    ############################################################################
    { # missing vary bottleneck size
        "procedure.database":"thesis_experiments_v2.db",
        "experiment.repetition":"0,1,2,3,4,5,6,7,8,9",
        "procedure.her.max_replays":"0",
        # "procedure.movement_noise_magnitude_limit":"0.5",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_bn",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"20,5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck) + H critic
    ############################################################################
    { # missing vary exploration ratio with bottleneck_size = 5
        "procedure.database":"thesis_experiments_v2.db",
        "experiment.repetition":"0",
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "procedure.primitive_noise_magnitude_limit":"0.01",
        # "procedure.movement_noise_magnitude_limit":"0.5",
        # "procedure.primitive_noise_magnitude_limit":"0.5",
        "agent.tau":"0.01",
        "procedure.collection":"hrl_smaller_noise",
        "agent/policy_model_arch":"movement_primitive_3_3",
        "agent.policy_bottleneck_size":"5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        # "agent.exploration_prob":"0.2",
        "agent.exploration_prob":"0.1",
        "agent.movement_exploration_prob_ratio":"0.0,0.2,0.5,0.8,1.0",
        "agent.movement_exploration_stddev":"0.5",
        "agent.primitive_exploration_stddev":"0.2",
    },
]






experiments = [
    ############################################################################
    # -- -- plain old TD3
    ############################################################################
    { # standard_td3_vary_exploration_prob with exploration std = 0.5
        "procedure.database":"thesis_experiments_v4.db",
        "experiment.repetition":"0,1,2,3,4",
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0,0.01",
        "agent.tau":"0.01",
        "procedure.collection":"standard_low_lr",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1,1.0",
        "agent.policy_movement_learning_rate":1e-4,
        "agent.policy_primitive_learning_rate":1e-4,
    },

    { # standard_td3_vary_exploration_prob with exploration std = 0.5
        "procedure.database":"thesis_experiments_v4.db",
        "experiment.repetition":"0,1,2,3,4",
        "procedure.her.max_replays":"0",
        "procedure.movement_noise_magnitude_limit":"0,0.01",
        "agent.tau":"0.01",
        "procedure.collection":"standard_shallow_net",
        "agent/policy_model_arch":"movement_3_layers",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1,1.0",
    },

    { # standard_td3_vary_exploration_prob with exploration std = 0.5
        "procedure.database":"thesis_experiments_v4.db",
        "experiment.repetition":"0,1,2,3,4,5,6,7,8,9",
        "procedure.her.max_replays":"3",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"standard_HER",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"0.2",
        "experiment.n_episodes":"100000",
        "agent.movement_exploration_stddev":"0.5",
        "agent.exploration_prob":"0.1",
    },

    ############################################################################
    # -- -- TD3 + action sequence (without bottleneck)
    ############################################################################
    { # missing vary sequence length with exploration std = 0.5 / exploration prob = 1.0
        "procedure.database":"thesis_experiments_v4.db",
        "experiment.repetition":"0,1,2,3,4,5,6,7,8,9",
        "procedure.her.max_replays":"3",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_HER",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"200",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck)
    ############################################################################
    { # missing vary bottleneck size
        "procedure.database":"thesis_experiments_v4.db",
        "experiment.repetition":"0,1,2,3,4,5,6,7,8,9",
        "procedure.her.max_replays":"3",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"trajectory_bn_HER",
        "agent/policy_model_arch":"movement_3_3",
        "agent.policy_bottleneck_size":"5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1",
        "agent.movement_exploration_stddev":"0.5",
    },

    ############################################################################
    # -- -- TD3 + action sequence (with bottleneck) + H critic
    ############################################################################
    { # missing vary exploration ratio with bottleneck_size = 5
        "procedure.database":"thesis_experiments_v4.db",
        "experiment.repetition":"0,1,2,3,4,5,6,7,8,9",
        "procedure.her.max_replays":"3",
        "procedure.movement_noise_magnitude_limit":"0.01",
        "procedure.primitive_noise_magnitude_limit":"0.01",
        "agent.tau":"0.01",
        "procedure.collection":"hrl_smaller_noise_HER",
        "agent/policy_model_arch":"movement_primitive_3_3",
        "agent.policy_bottleneck_size":"5",
        "procedure.movement_mode":"full_raw",
        "procedure.movement_span_in_sec":"1.0",
        "experiment.n_episodes":"100000",
        "agent.exploration_prob":"0.1",
        "agent.movement_exploration_prob_ratio":"0.8",
        "agent.movement_exploration_stddev":"0.5",
        "agent.primitive_exploration_stddev":"0.2",
    },
]



# for i in range(5):
run_experiment(delay='now', **experiments[0])

# run_experiment(delay=1, **experiments[1])
# run_experiment(delay=3, **experiments[2])
# run_experiment(delay=4, **experiments[3])
# run_experiment(delay=5, **experiments[4])
# run_experiment(delay=6, **experiments[5])

# for i in range(3):
#     for experiment in experiments:
#         run_experiment(delay=i * 2, **experiment)

# for i in range(1):
#     for experiment in experiments:
#         run_experiment(delay='now', **experiment)

# run_experiment(delay=0, **experiments[1])


# n_exp = 0
# for experiment in experiments:
#     n_subexp = 1
#     for val in experiment.values():
#             n_subexp *= len(val.split(","))
#     print(experiment["procedure.collection"], n_subexp)
#     n_exp += n_subexp
# print("total", n_exp)

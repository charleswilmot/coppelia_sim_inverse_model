# one action only, no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=full_raw procedure.movement_span_in_sec=0.2 agent/policy_model_arch=default


# one action only, movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=full_raw procedure.movement_span_in_sec=0.2 agent/policy_model_arch=movement_primitive_3_3


# one bell only, no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=minimalist agent/policy_model_arch=default


# one bell only, movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=minimalist agent/policy_model_arch=movement_primitive_3_3


# one cubic_hermite only, no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=cubic_hermite agent/policy_model_arch=default


# one cubic_hermite only, movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=cubic_hermite agent/policy_model_arch=movement_primitive_3_3


# many actions (10 / 2sec), no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=full_raw agent/policy_model_arch=default


# many actions (10 / 2sec), movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap experiment.n_episodes=30000 procedure.movement_mode=full_raw agent/policy_model_arch=movement_primitive_3_3

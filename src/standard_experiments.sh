# one action only, no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=full_raw procedure.movement_span_in_sec=0.2 agent/policy_model_arch=default


# one action only, movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=full_raw procedure.movement_span_in_sec=0.2 agent/policy_model_arch=movement_primitive_3_3 procedure.overwrite_movement_discount_factor=0.7


# one bell only, no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=minimalist agent/policy_model_arch=default


# one bell only, movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=minimalist agent/policy_model_arch=movement_primitive_3_3 procedure.overwrite_movement_discount_factor=0.7


# many actions (10 / 2sec), no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=full_raw agent/policy_model_arch=default


# many actions (10 / 2sec), movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=full_raw agent/policy_model_arch=movement_primitive_3_3 procedure.overwrite_movement_discount_factor=0.7 procedure.movement_span_in_sec=1.2 agent.policy_movement_learning_rate=8e-5








###################################################################################################################################################################################################################################

# one cubic_hermite only, no movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=cubic_hermite agent/policy_model_arch=default


# one cubic_hermite only, movement primitive
python cluster.py --multirun experiment.repetition=0,1,2 simulation.environment=one_arm_4_buttons,one_arm_2_buttons_1_levers_1_tap procedure.movement_mode=cubic_hermite agent/policy_model_arch=movement_primitive_3_3

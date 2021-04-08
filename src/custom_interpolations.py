import numpy as np
from omegaconf import OmegaConf


def get_policy_output_size(movement_mode, simulation_timestep, movement_span_in_sec, n_joints):
    simulation_timestep = float(simulation_timestep)
    movement_span_in_sec = float(movement_span_in_sec)
    n_joints = int(n_joints)
    if movement_mode in ["full_raw"]:
        n_actions = int(movement_span_in_sec / simulation_timestep)
        policy_output_size = n_actions * n_joints
    elif movement_mode in ["minimalist", "one_raw"]:
        policy_output_size = n_joints
    elif movement_mode in ["cubic_hermite"]:
        policy_output_size = n_joints * 4
    else:
        raise ValueError("Movement mode not recognized ({})".format(movement_mode))
    return policy_output_size


def get_action_size(movement_mode, n_joints):
    n_joints = int(n_joints)
    if movement_mode in ["full_raw", "minimalist", "one_raw"]:
        return n_joints
    elif movement_mode in ["cubic_hermite"]:
        return n_joints * 4
    else:
        raise ValueError("Movement mode not recognized ({})".format(movement_mode))


OmegaConf.register_new_resolver("get_policy_output_size", get_policy_output_size)
OmegaConf.register_new_resolver("get_action_size", get_action_size)
OmegaConf.register_new_resolver("slash_to_dot", lambda x: x.replace('/', '.').replace('agent', 'a').replace('experiment', 'e').replace('procedure', 'p')[:235])
OmegaConf.register_new_resolver("log", lambda x: np.log(float(x)))
OmegaConf.register_new_resolver("sum", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("pad_number", lambda x: "{:02d}".format(int(x)))

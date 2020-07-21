from procedure import Procedure
from omegaconf import OmegaConf
from pathlib import Path
import hydra
import sys


def get_conf():
    cfg = OmegaConf.load('../.hydra/config.yaml')
    return cfg


@hydra.main(config_path="../config/replay.yml", strict=True)
def replay(cfg):
    experiment_cfg = get_conf()
    agent_conf = experiment_cfg.agent
    policy_buffer_conf = experiment_cfg.policy_buffer
    critic_buffer_conf = experiment_cfg.critic_buffer
    simulation_conf = experiment_cfg.simulation
    procedure_conf = experiment_cfg.procedure
    simulation_conf.n = 1
    if cfg.gui:
        simulation_conf.guis = [0]

    video_name = 'replay_exploration.mp4' if cfg.exploration else 'replay.mp4'
    relative_checkpoint_path = "../checkpoints/" + Path(cfg.path).stem
    with Procedure(agent_conf, policy_buffer_conf, critic_buffer_conf,
            simulation_conf, procedure_conf) as procedure:
        procedure.restore(relative_checkpoint_path)
        procedure.replay(
            exploration=cfg.exploration,
            record=cfg.record,
            n_episodes=cfg.n_episodes,
            resolution=cfg.resolution,
            video_name=video_name
        )


if __name__ == '__main__':
    replay()

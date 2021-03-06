from procedure import Procedure
from omegaconf import OmegaConf
from pathlib import Path
import hydra
import sys
import custom_interpolations


def get_conf():
    cfg = OmegaConf.load('../.hydra/config.yaml')
    return cfg


@hydra.main(config_path="../config/replay/", config_name="replay.yaml")
def main(cfg):
    replay(cfg)


def replay(cfg):
    experiment_cfg = get_conf()
    agent_conf = experiment_cfg.agent
    buffer_conf = experiment_cfg.buffer
    simulation_conf = experiment_cfg.simulation
    procedure_conf = experiment_cfg.procedure
    # simulation_conf.n = cfg.n_overlays if cfg.overlay else 1
    if cfg.gui:
        simulation_conf.guis = [0]

    goals_pairs = [
        ([1, 0, 0, 0], [0, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 1, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 0, 0]),
        ([0, 1, 1, 1], [1, 1, 1, 1]),
        ([1, 0, 1, 1], [1, 1, 1, 1]),
        ([1, 1, 0, 1], [1, 1, 1, 1]),
        ([1, 1, 1, 0], [1, 1, 1, 1]),
    ]


    video_name = 'replay_exploration' if cfg.exploration else 'replay'
    relative_checkpoint_path = "../checkpoints/" + Path(cfg.path).stem
    with Procedure(agent_conf, buffer_conf, simulation_conf,
            procedure_conf) as procedure:
        procedure.restore(relative_checkpoint_path)
        if cfg.overlay:
            procedure.replay_exploration_overlay(
                goals_pairs=goals_pairs,
                std=cfg.std,
                video_name='overlay_exploration_{}.mp4'.format(cfg.std),
                resolution=cfg.resolution,
            )
        else:
            procedure.replay(
                exploration=cfg.exploration,
                record=cfg.record,
                n_episodes=cfg.n_episodes,
                resolution=cfg.resolution,
                video_name=video_name
            )


if __name__ == '__main__':
    main()

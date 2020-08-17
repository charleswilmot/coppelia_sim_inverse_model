from experiment import experiment
import hydra
import json
import omegaconf
import os


@hydra.main(config_path="../config/cluster", config_name='cluster.yaml')
def remote_experiment(cfg):
    os.chdir(cfg.rundir)
    cfg = omegaconf.OmegaConf.load('./cfg.yaml')
    experiment(cfg)


if __name__ == '__main__':
    remote_experiment()

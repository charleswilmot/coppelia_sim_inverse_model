from experiment import experiment
import hydra
import json
import omegaconf
import os


@hydra.main(config_path="../config/cluster.yml", strict=True)
def remote_experiment(cfg):
    with open(cfg.rundir + '/cfg.json', 'r') as f:
        other_cfg_json = json.load(f)
    os.chdir(cfg.rundir)
    cfg = omegaconf.OmegaConf.create(other_cfg_json)
    experiment(cfg)


if __name__ == '__main__':
    remote_experiment()

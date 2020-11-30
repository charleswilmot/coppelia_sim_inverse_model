from replay import replay
import hydra
import json
import omegaconf
import os
import custom_interpolations


@hydra.main(config_path="../config/cluster/cluster.yaml", strict=True)
def remote_record(cfg):
    with open(cfg.rundir + '/cfg.json', 'r') as f:
        other_cfg_json = json.load(f)
    os.chdir(cfg.rundir)
    cfg = omegaconf.OmegaConf.create(other_cfg_json)
    replay(cfg)


if __name__ == '__main__':
    remote_record()

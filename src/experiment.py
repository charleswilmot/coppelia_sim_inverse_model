import hydra
import omegaconf
from procedure import Procedure
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@hydra.main(config_path='../config/general', config_name='default.yaml')
def main(cfg):
    experiment(cfg)


def experiment(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg), end="\n\n\n")
    agent_conf = cfg.agent
    buffer_conf = cfg.buffer
    simulation_conf = cfg.simulation
    procedure_conf = cfg.procedure
    experiment_conf = cfg.experiment
    with Procedure(agent_conf, buffer_conf, simulation_conf, procedure_conf) as procedure:
        n_episode_batch = experiment_conf.n_episodes // simulation_conf.n
        for episode_batch in range(n_episode_batch):
            policy = (episode_batch + 1) % experiment_conf.policy_every == 0
            critic = (episode_batch + 1) % experiment_conf.critic_every == 0
            forward = (episode_batch + 1) % experiment_conf.forward_every == 0
            evaluation = (episode_batch + 1) % experiment_conf.evaluate_every == 0
            save = (episode_batch + 1) % experiment_conf.save_every == 0
            record = (episode_batch + 1) % experiment_conf.record_episode_every == 0
            dump_buffers = episode_batch in experiment_conf.dump_buffers_at
            print_info = (episode_batch + 1) % 10 == 0
            print("batch {: 5d}\tevaluation:{}\tpolicy:{}\tcritic:{}\tforward:{}\tsave:{}\trecord:{}\tdump buffers:{}".format(
                episode_batch + 1,
                evaluation,
                policy,
                critic,
                forward,
                save,
                record,
                dump_buffers
            ))
            procedure.collect_train_and_log(policy=policy, critic=critic, forward=forward, evaluation=evaluation)
            if save:
                procedure.save()
            if record:
                procedure.replay(
                    record=True,
                    video_name='./replays/replay_{:05d}.mp4'.format(episode_batch),
                    n_episodes=1,
                    exploration=False
                )
            if dump_buffers:
                procedure.dump_buffers()
            if print_info:
                print('n_exploration_episodes  ...  ', procedure.n_exploration_episodes)
                print('n_evaluation_episodes  ....  ', procedure.n_evaluation_episodes)
                print('n_transition_gathered  ....  ', procedure.n_transition_gathered)
                print('n_policy_training  ........  ', procedure.n_policy_training)
                print('n_critic_training  ........  ', procedure.n_critic_training)
                print('n_forward_training  .......  ', procedure.n_forward_training)
                print('n_global_training  ........  ', procedure.n_global_training)
        if not save:
            procedure.save()
        if experiment_conf.final_recording:
            print("Generating final recording (without exploration)")
            procedure.replay(
                record=True,
                video_name='./replays/replay_final.mp4',
                n_episodes=10,
                exploration=False
            )
        print("Experiment finished, hope it worked. Good bye!")


if __name__ == "__main__":
    main()

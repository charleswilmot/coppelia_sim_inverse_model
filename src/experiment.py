import hydra
from procedure import Procedure


@hydra.main(config_path='../config/general/default.yaml', strict=True)
def main(cfg):
    experiment(cfg)


def experiment(cfg):
    print(cfg.pretty(), end="\n\n\n")
    agent_conf = cfg.agent
    policy_buffer_conf = cfg.policy_buffer
    critic_buffer_conf = cfg.critic_buffer
    simulation_conf = cfg.simulation
    procedure_conf = cfg.procedure
    experiment_conf = cfg.experiment
    with Procedure(agent_conf, policy_buffer_conf, critic_buffer_conf,
            simulation_conf, procedure_conf) as procedure:
        n_episode_batch = experiment_conf.n_episodes // simulation_conf.n
        for episode_batch in range(n_episode_batch):
            policy = episode_batch % experiment_conf.policy_every == 0
            critic = episode_batch % experiment_conf.critic_every == 0
            save = (episode_batch + 1) % experiment_conf.save_every == 0
            record = (episode_batch + 1) % experiment_conf.record_episode_every == 0
            print_info = (episode_batch + 1) % 10 == 0
            print("batch {: 5d}\tpolicy:{}\tcritic:{}\tsave:{}\trecord:{}".format(
                episode_batch,
                policy,
                critic,
                save,
                record
            ))
            procedure.gather_train_and_log(policy=policy, critic=critic)
            if save:
                procedure.save()
            if record:
                procedure.replay(
                    record=True,
                    video_name='./replays/replay_{:05d}.mp4'.format(episode_batch),
                    n_episodes=1
                )
            if print_info:
                print('n_policy_transition_gathered...', procedure.n_policy_transition_gathered)
                print('n_policy_training..............', procedure.n_policy_training)
                print('current_policy_ratio...........', procedure.current_policy_ratio)
                print('n_critic_transition_gathered...', procedure.n_critic_transition_gathered)
                print('n_critic_training..............', procedure.n_critic_training)
                print('current_critic_ratio...........', procedure.current_critic_ratio)
        if not save:
            procedure.save()
        if experiment_conf.final_recording:
            print("Generating final recording (without exploration)")
            procedure.replay(
                record=True,
                video_name='./replays/replay_final.mp4',
                n_episodes=10,
                explore=False
            )
            print("Generating final recording (with exploration)")
            procedure.replay(
                record=True,
                video_name='./replays/replay_final_exploration.mp4',
                n_episodes=10,
                explore=True
            )
        print("Experiment finished, hope it worked. Good bye!")


if __name__ == "__main__":
    main()

import hydra
from procedure import Procedure


@hydra.main(config_path='../config/default.yml', strict=True)
def main(cfg):
    print(cfg.pretty(), end="\n\n\n")
    agent_conf = cfg.agent
    policy_buffer_conf = cfg.policy_buffer
    critic_buffer_conf = cfg.critic_buffer
    simulation_conf = cfg.simulation
    procedure_conf = cfg.procedure
    with Procedure(agent_conf, policy_buffer_conf, critic_buffer_conf,
            simulation_conf, procedure_conf) as procedure:
        # procedure.gather_and_train_critic()
        for i in range(50):
            print('critic only', i)
            procedure.gather_train_and_log(policy=False)
        for i in range(50, 1000):
            print('both', i)
            procedure.gather_train_and_log()
        print("n_policy_episodes", procedure.n_policy_episodes)
        print("n_critic_episodes", procedure.n_critic_episodes)
        print("n_policy_transition_gathered", procedure.n_policy_transition_gathered)
        print("n_critic_transition_gathered", procedure.n_critic_transition_gathered)
        print("n_policy_training", procedure.n_policy_training)
        print("n_critic_training", procedure.n_critic_training)
        print("current_policy_ratio", procedure.current_policy_ratio)
        print("current_critic_ratio", procedure.current_critic_ratio)


if __name__ == "__main__":
    main()

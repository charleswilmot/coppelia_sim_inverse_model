

if __name__ == '__main__':
    from database import Database
    db = Database('../databases/version_1.db')



    # experiment_ids = db.get_experiment_ids(collection='standard_td3_vary_exploration_std', movement_exploration_stddev=0.5)
    # new_collection = 'standard_td3_vary_exploration_prob'

    # experiment_ids = db.get_experiment_ids(collection='standard_td3_vary_exploration_prob', exploration_prob=1.0)
    # new_collection = 'standard_td3_vary_exploration_std'

    # experiment_ids = db.get_experiment_ids(collection='standard_td3_vary_exploration_prob', exploration_prob=0.55)
    # new_collection = 'standard_td3_no_ada_step_vary_tau'

    # experiment_ids = db.get_experiment_ids(collection='standard_td3_no_ada_step_vary_tau', tau=0.01)
    # new_collection = 'standard_td3_vary_exploration_prob'

    # experiment_ids = db.get_experiment_ids(collection='standard_td3_vary_exploration_prob_ada_step', exploration_prob=0.55)
    # new_collection = 'standard_td3_ada_step_vary_tau'

    # experiment_ids = db.get_experiment_ids(collection='standard_td3_ada_step_vary_tau', tau=0.01)
    # new_collection = 'standard_td3_vary_exploration_prob_ada_step'

    # experiment_ids = db.get_experiment_ids(collection='standard_td3_vary_HER', her_max_replays=2)
    # new_collection = 'trajectory_td3_vary_traj_len'

    # experiment_ids = db.get_experiment_ids(collection='trajectory_td3_vary_traj_len', n_actions_in_movement=1)
    # new_collection = 'standard_td3_vary_HER'

    # experiment_ids = db.get_experiment_ids(collection='trajectory_td3_vary_traj_len', n_actions_in_movement=5)
    # new_collection = 'trajectory_td3_vary_timestep'

    # experiment_ids = db.get_experiment_ids(collection='trajectory_td3_vary_traj_len', n_actions_in_movement=1)
    # new_collection = 'standard_td3_vary_traj_len'

    # UPDATE experiments SET collection='standard_td3_vary_timestep' WHERE collection='standard_td3_vary_traj_len';


    df = db.get_dataframe('SELECT finished FROM experiments WHERE experiment_id IN {}'.format(tuple(experiment_ids)))
    if not (df.finished.values == 1).all():
        raise ValueError('Some of the experiments are not completed!')
    res = input('Do you want to copy the following experiments into the collection  {}? (yes)  {}'.format(new_collection, experiment_ids))
    if res == 'yes':
        for experiment_id in experiment_ids:
            db.copy_to_new_collection(experiment_id, new_collection)
    elif res == 'no':
        print('discarding')
    else:
        experiment_ids = [int(x) for x in res.split(',')]
        res = input('Do you want to copy the following experiments into the collection  {}? (yes)  {}'.format(new_collection, experiment_ids))
        if res == 'yes':
            for experiment_id in experiment_ids:
                db.copy_to_new_collection(experiment_id, new_collection)

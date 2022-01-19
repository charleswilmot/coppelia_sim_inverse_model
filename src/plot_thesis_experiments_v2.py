import os
from shutil import copy
from database import Database
from plot_from_db import plot_1_variable_param, plot_collection_metric, get_average_perf
import matplotlib.pyplot as plt


def plot_db_thesis_experiments_v2():
    plt.rc('legend',fontsize=4)

    db = Database('../databases/thesis_experiments_v3.db')
    smoothing = 2

    collections_at = (
        (
            "standard_wrt_exploration_prob_nml_0",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "standard_wrt_exploration_prob_nml_0.01",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        # (
        #     "standard_wrt_exploration_prob_nml_0.5",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "standard_wrt_nml_exploration_prob_0.1",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "standard_wrt_nml_exploration_prob_1.0",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "bell_wrt_exploration_prob_nml_0",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "bell_wrt_exploration_prob_nml_0.01",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "bell_wrt_exploration_prob_nml_0.5",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "bell_wrt_nml_exploration_prob_0.1",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "bell_wrt_nml_exploration_prob_1.0",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        (
            "trajectory_wrt_n_actions_in_movement_nml_0",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "trajectory_wrt_n_actions_in_movement_nml_0.01",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        # (
        #     "trajectory_wrt_n_actions_in_movement_nml_0.5",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "trajectory_wrt_nml_n_actions_in_movement_1",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "trajectory_wrt_nml_n_actions_in_movement_3",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "trajectory_wrt_nml_n_actions_in_movement_5",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "trajectory_wrt_nml_n_actions_in_movement_6",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        (
            "trajectory_bn_wrt_bn_nml0.01",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        # (
        #     "trajectory_bn_wrt_bn_nml0.5",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "hrl_wrt_exploration_prob_ratio_nml0.01",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        # (
        #     "hrl_wrt_exploration_prob_ratio_nml0.5",
        #     [10400, 20000, 40000, 60000, 80000, 98400],
        #     None,
        # ),
        (
            "hrl_smaller_noise_wrt_exploration_prob_ratio_nml0.01",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
    )

    for i, (collection, at, keep) in enumerate(collections_at):
        if i > -1:
            for reduction_method in ['mean_std']: #, 'mean_std_2', 'mean_std_5']: #, 'median', 'max']:
                print('{}_{:02d}_{}'.format(reduction_method, i, collection))
                fig = plt.figure()
                log = plot_1_variable_param(fig, db, collection, at, keep, smoothing=smoothing, reduction_method=reduction_method)
                fig.savefig('../plots_3/{}_{:02d}_{}.png'.format(reduction_method, i, collection), dpi=300)
                with open('../plots_3/{}_{:02d}_{}.txt'.format(reduction_method, i, collection), "w") as f:
                    f.write('# {}_{:02d}_{}.png\n'.format(reduction_method, i, collection))
                    for value, n_episodes in log.items():
                        f.write(f"{value} -- {len(n_episodes)} {n_episodes}\n")
                    f.write('\n')
                plt.close(fig)




def plot_db_thesis_experiments_vfinal():
    try:
        os.remove("../databases/thesis_experiments_v3.db")
    except OSError:
        pass
    copy("../databases/thesis_experiments_v2.db", "../databases/thesis_experiments_v3.db")


    db_1 = Database('../databases/thesis_experiments.db')
    db_2 = Database('../databases/thesis_experiments_v2.db')
    db_3 = Database('../databases/thesis_experiments_v3.db')

    # import matching experiments from 'thesis_experiments.db'
    db_2_groups = db_2.group_by_matching_experiments()
    for group in db_2_groups:
        matchings = db_2.get_matching_experiments_ids(db_1, group[0])
        collection = db_2.get_experiment_field(group[0], 'collection')
        # print(f"db_2's group {group} ({collection}) matches with {matchings} in db_1")
        for matching in matchings:
            db_3.import_from_other_db(db_1, matching, collection)


    print("\n" * 10)
    print("Creating the plotting collections:")
    # create the plotting collections
    db_3_groups = db_3.group_by_matching_experiments()
    db_3.get_experiment_ids()

    # - collection standard, wrt exploration_prob
    for nml in [0, 0.01, 0.5]:
        originals = db_3.get_experiment_ids(collection='standard', movement_noise_magnitude_limit=nml)
        to_copy = set(originals)
        for original in originals:
            for group in db_3_groups:
                if original in group:
                    for group_member in group:
                        to_copy.add(group_member)
        to_copy = list(sorted(to_copy))
        collection = f'standard_wrt_exploration_prob_nml_{nml}'
        print(f'nml={nml}, will copy {to_copy} into collection {collection}')
        for experiment_id in to_copy:
            db_3.copy_to_new_collection(experiment_id, collection)

    # - collection standard, wrt movement_noise_magnitude_limit
    for exploration_prob in [0.1, 1.0]:
        originals = db_3.get_experiment_ids(collection='standard', exploration_prob=exploration_prob)
        to_copy = set(originals)
        for original in originals:
            for group in db_3_groups:
                if original in group:
                    for group_member in group:
                        to_copy.add(group_member)
        to_copy = list(sorted(to_copy))
        collection = f'standard_wrt_nml_exploration_prob_{exploration_prob}'
        print(f'exploration_prob={exploration_prob}, will copy {to_copy} into collection {collection}')
        for experiment_id in to_copy:
            db_3.copy_to_new_collection(experiment_id, collection)

    # - collection bell, wrt exploration_prob
    for nml in [0, 0.01, 0.5]:
        originals = db_3.get_experiment_ids(collection='bell', movement_noise_magnitude_limit=nml)
        to_copy = set(originals)
        for original in originals:
            for group in db_3_groups:
                if original in group:
                    for group_member in group:
                        to_copy.add(group_member)
        to_copy = list(sorted(to_copy))
        collection = f'bell_wrt_exploration_prob_nml_{nml}'
        print(f'nml={nml}, will copy {to_copy} into collection {collection}')
        for experiment_id in to_copy:
            db_3.copy_to_new_collection(experiment_id, collection)

    # - collection bell, wrt movement_noise_magnitude_limit
    for exploration_prob in [0.1, 1.0]:
        originals = db_3.get_experiment_ids(collection='bell', exploration_prob=exploration_prob)
        to_copy = set(originals)
        for original in originals:
            for group in db_3_groups:
                if original in group:
                    for group_member in group:
                        to_copy.add(group_member)
        to_copy = list(sorted(to_copy))
        collection = f'bell_wrt_nml_exploration_prob_{exploration_prob}'
        print(f'exploration_prob={exploration_prob}, will copy {to_copy} into collection {collection}')
        for experiment_id in to_copy:
            db_3.copy_to_new_collection(experiment_id, collection)


    # - collection trajectory, wrt n_actions_in_movement
    for nml in [0, 0.01, 0.5]:
        originals = db_3.get_experiment_ids(collection='trajectory', movement_noise_magnitude_limit=nml)
        to_copy = set(originals)
        for original in originals:
            for group in db_3_groups:
                if original in group:
                    for group_member in group:
                        to_copy.add(group_member)
        to_copy = list(sorted(to_copy))
        collection = f'trajectory_wrt_n_actions_in_movement_nml_{nml}'
        print(f'nml={nml}, will copy {to_copy} into collection {collection}')
        for experiment_id in to_copy:
            db_3.copy_to_new_collection(experiment_id, collection)

    # - collection trajectory, wrt movement_noise_magnitude_limit
    for n_actions_in_movement in [1, 3, 5, 6]:
        originals = db_3.get_experiment_ids(collection='trajectory', n_actions_in_movement=n_actions_in_movement)
        to_copy = set(originals)
        for original in originals:
            for group in db_3_groups:
                if original in group:
                    for group_member in group:
                        to_copy.add(group_member)
        to_copy = list(sorted(to_copy))
        collection = f'trajectory_wrt_nml_n_actions_in_movement_{n_actions_in_movement}'
        print(f'n_actions_in_movement={n_actions_in_movement}, will copy {to_copy} into collection {collection}')
        for experiment_id in to_copy:
            db_3.copy_to_new_collection(experiment_id, collection)

    # - collection trajectory_bn, wrt bottleneck_size
    originals = (
        *db_3.get_experiment_ids(
            collection='trajectory_bn',
            movement_noise_magnitude_limit=0.5),
        *db_3.get_experiment_ids(
            collection='trajectory',
            n_actions_in_movement=5,
            movement_noise_magnitude_limit=0.5),
    )
    to_copy = set(originals)
    for original in originals:
        for group in db_3_groups:
            if original in group:
                for group_member in group:
                    to_copy.add(group_member)
    to_copy = list(sorted(to_copy))

    print(f"trajectory_bn, will copy {to_copy} into collection 'trajectory_bn_wrt_bn_nml0.5'")
    for experiment_id in to_copy:
        db_3.copy_to_new_collection(experiment_id, 'trajectory_bn_wrt_bn_nml0.5')

    originals = (
        *db_3.get_experiment_ids(
            collection='trajectory_bn',
            movement_noise_magnitude_limit=0.01),
        *db_3.get_experiment_ids(
            collection='trajectory',
            n_actions_in_movement=5,
            movement_noise_magnitude_limit=0.01),
    )
    to_copy = set(originals)
    for original in originals:
        for group in db_3_groups:
            if original in group:
                for group_member in group:
                    to_copy.add(group_member)
    to_copy = list(sorted(to_copy))

    print(f"trajectory_bn, will copy {to_copy} into collection 'trajectory_bn_wrt_bn_nml0.01'")
    for experiment_id in to_copy:
        db_3.copy_to_new_collection(experiment_id, 'trajectory_bn_wrt_bn_nml0.01')


    # - collection hrl, wrt movement_exploration_prob_ratio
    originals = db_3.get_experiment_ids(
        collection='hrl',
        movement_noise_magnitude_limit=0.01)
    to_copy = set(originals)
    for original in originals:
        for group in db_3_groups:
            if original in group:
                for group_member in group:
                    to_copy.add(group_member)
    to_copy = list(sorted(to_copy))

    print(f"hrl, will copy {to_copy} into collection 'hrl_wrt_exploration_prob_ratio_nml0.01'")
    for experiment_id in to_copy:
        db_3.copy_to_new_collection(experiment_id, 'hrl_wrt_exploration_prob_ratio_nml0.01')

    originals = db_3.get_experiment_ids(
        collection='hrl',
        movement_noise_magnitude_limit=0.5)
    to_copy = set(originals)
    for original in originals:
        for group in db_3_groups:
            if original in group:
                for group_member in group:
                    to_copy.add(group_member)
    to_copy = list(sorted(to_copy))

    print(f"hrl, will copy {to_copy} into collection 'hrl_wrt_exploration_prob_ratio_nml0.5'")
    for experiment_id in to_copy:
        db_3.copy_to_new_collection(experiment_id, 'hrl_wrt_exploration_prob_ratio_nml0.5')



    originals = db_3.get_experiment_ids(
        collection='hrl_smaller_noise',
        movement_noise_magnitude_limit=0.01)
    to_copy = set(originals)
    for original in originals:
        for group in db_3_groups:
            if original in group:
                for group_member in group:
                    to_copy.add(group_member)
    to_copy = list(sorted(to_copy))

    print(f"hrl, will copy {to_copy} into collection 'hrl_smaller_noise_wrt_exploration_prob_ratio_nml0.01'")
    for experiment_id in to_copy:
        db_3.copy_to_new_collection(experiment_id, 'hrl_smaller_noise_wrt_exploration_prob_ratio_nml0.01')



    # Display all groups after the transactions
    print("\n" * 10)

    db_3_groups = db_3.group_by_matching_experiments()
    for group in db_3_groups:
        print(f'These experiments match: {group}')
        for experiment_id in group:
            # df = db_3.get_dataframe(f"SELECT collection FROM experiments WHERE experiment_id='{experiment_id}'")
            # collection = df.collection.values[0]
            collection = db_3.get_experiment_field(experiment_id, 'collection')
            print(experiment_id, collection)
        print("")


    # plotting
    print("\n" * 10)
    print("plotting")

    # plt.rc('legend',fontsize=14)

    db = Database('../databases/thesis_experiments_v3.db')
    smoothing = 2

    collections = [
        "standard_wrt_exploration_prob_nml_0",
        "standard_wrt_exploration_prob_nml_0.01",
        "trajectory_wrt_n_actions_in_movement_nml_0",
        "trajectory_wrt_n_actions_in_movement_nml_0.01",
        "trajectory_bn_wrt_bn_nml0.01",
        "hrl_smaller_noise_wrt_exploration_prob_ratio_nml0.01",
    ]
    arerage_perf = {
        collection: get_average_perf(db, 'success_rate_percent', collection)
        for collection in collections
    }

    i = 0
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "standard_wrt_exploration_prob_nml_0", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"$1$-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "standard_wrt_exploration_prob_nml_0"), dpi=300)
    plt.close(fig)

    i = 1
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "standard_wrt_exploration_prob_nml_0.01", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"max-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "standard_wrt_exploration_prob_nml_0.01"), dpi=300)
    plt.close(fig)

    i = 2
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "trajectory_wrt_n_actions_in_movement_nml_0", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$n=$')
    ax.set_title(r"$1$-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "trajectory_wrt_n_actions_in_movement_nml_0"), dpi=300)
    plt.close(fig)

    i = 3
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "trajectory_wrt_n_actions_in_movement_nml_0.01", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$n=$')
    ax.set_title(r"max-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "trajectory_wrt_n_actions_in_movement_nml_0.01"), dpi=300)
    plt.close(fig)

    i = 4
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "trajectory_bn_wrt_bn_nml0.01", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$|z|=$')
    ax.set_title(r"with bottleneck")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "trajectory_bn_wrt_bn_nml0.01"), dpi=300)
    plt.close(fig)

    i = 5
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "hrl_smaller_noise_wrt_exploration_prob_ratio_nml0.01", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_{\mathrm{explore},H}=$')
    ax.set_title(r"HRL")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "hrl_smaller_noise_wrt_exploration_prob_ratio_nml0.01"), dpi=300)
    plt.close(fig)




    i = 6
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "standard_wrt_exploration_prob_nml_0", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$', above=12)
    ax.set_title(r"$1$-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "standard_wrt_exploration_prob_nml_0"), dpi=300)
    plt.close(fig)

    i = 7
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db, "success_rate_percent", "standard_wrt_exploration_prob_nml_0.01", smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$', above=12)
    ax.set_title(r"max-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_4/{:02d}_{}.png'.format(i, "standard_wrt_exploration_prob_nml_0.01"), dpi=300)
    plt.close(fig)


def plot_db_thesis_experiments_v4():
    try:
        os.remove("../databases/thesis_experiments_v5.db")
    except OSError:
        pass
    copy("../databases/thesis_experiments_v4.db", "../databases/thesis_experiments_v5.db")


    db_1 = Database('../databases/thesis_experiments_v4.db')
    db_2 = Database('../databases/thesis_experiments_v5.db')

    experiment_ids = db_1.get_experiment_ids(collection='standard_low_lr', movement_noise_magnitude_limit=0)
    for experiment_id in experiment_ids:
        db_2.copy_to_new_collection(experiment_id, "standard_low_lr_nml_0")
    experiment_ids = db_1.get_experiment_ids(collection='standard_low_lr', movement_noise_magnitude_limit=0.01)
    for experiment_id in experiment_ids:
        db_2.copy_to_new_collection(experiment_id, "standard_low_lr_nml_0.01")
    experiment_ids = db_1.get_experiment_ids(collection='standard_shallow_net', movement_noise_magnitude_limit=0)
    for experiment_id in experiment_ids:
        db_2.copy_to_new_collection(experiment_id, "standard_shallow_net_nml_0")
    experiment_ids = db_1.get_experiment_ids(collection='standard_shallow_net', movement_noise_magnitude_limit=0.01)
    for experiment_id in experiment_ids:
        db_2.copy_to_new_collection(experiment_id, "standard_shallow_net_nml_0.01")

    smoothing = 2

    collections = [
        "standard_low_lr_nml_0",
        "standard_low_lr_nml_0.01",
        "standard_shallow_net_nml_0",
        "standard_shallow_net_nml_0.01",
        # "standard_HER",
        # "trajectory_HER",
        # "trajectory_bn_HER",
        # "hrl_smaller_noise_HER",
    ]
    arerage_perf = {
        collection: get_average_perf(db_2, 'success_rate_percent', collection)
        for collection in collections
    }

    i = 0
    collection = "standard_low_lr_nml_0"
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db_2, "success_rate_percent", collection, smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"$1$-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_5/{:02d}_{}.png'.format(i, collection), dpi=300)
    plt.close(fig)

    i = 1
    collection = "standard_low_lr_nml_0.01"
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db_2, "success_rate_percent", collection, smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"max-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_5/{:02d}_{}.png'.format(i, collection), dpi=300)
    plt.close(fig)

    i = 2
    collection = "standard_shallow_net_nml_0"
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db_2, "success_rate_percent", collection, smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"$1$-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_5/{:02d}_{}.png'.format(i, collection), dpi=300)
    plt.close(fig)

    i = 3
    collection = "standard_shallow_net_nml_0.01"
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db_2, "success_rate_percent", collection, smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"max-step TD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_5/{:02d}_{}.png'.format(i, collection), dpi=300)
    plt.close(fig)

    i = 4
    collection = "standard_HER"
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db_2, "success_rate_percent", collection, smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"max-step TD, HER")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_5/{:02d}_{}.png'.format(i, collection), dpi=300)
    plt.close(fig)

    i = 5
    collection = "trajectory_HER"
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    log = plot_collection_metric(ax, db_2, "success_rate_percent", collection, smoothing=smoothing, reduction_method="mean_std", ylim=[-5, 105], ylabel=r'success rate (%)', legend_prefix=r'$p_\mathrm{explore}=$')
    ax.set_title(r"max-step TD, HER")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('../plots_5/{:02d}_{}.png'.format(i, collection), dpi=300)
    plt.close(fig)




if __name__ == '__main__':

    # plot_db_thesis_experiments_vfinal()
    plot_db_thesis_experiments_v4()

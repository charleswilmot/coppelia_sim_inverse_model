from database import Database
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import namedtuple


SmoothedMetric = namedtuple('SmoothedMetric', ["x", "means", "stds", "medians", "maxs"])

start_color = np.array([170, 255, 255], dtype=np.float32) / 255
stop_color = np.array([000, 000, 110], dtype=np.float32) / 255
# start_color = np.array([255, 0, 0], dtype=np.float32) / 255
# stop_color = np.array([0, 255, 255], dtype=np.float32) / 255


def get_nlargest_smoothed_metric(db, experiment_ids, metric, smoothing=0, n=5):
    experiment_ids_str = str(tuple(experiment_ids)) if len(experiment_ids) > 1 else f"({experiment_ids[0]})"
    results = db.get_dataframe(f'''
    SELECT
        experiment_id,
        {metric}
    FROM
        results
    WHERE
        experiment_id IN {experiment_ids_str}
    ''').fillna(np.nan)
    nlargest_experiment_ids = results.groupby(['experiment_id']).mean().nlargest(n, metric).index.to_list()
    return get_smoothed_metric(db, nlargest_experiment_ids, metric, smoothing)


def get_smoothed_metric(db, experiment_ids, metric, smoothing=0):
    experiment_ids_str = str(tuple(experiment_ids)) if len(experiment_ids) > 1 else f"({experiment_ids[0]})"
    results = db.get_dataframe(f'''
    SELECT
        episode_nb,
        {metric}
    FROM
        results
    WHERE
        experiment_id IN {experiment_ids_str}
    ''').fillna(np.nan)
    results = results.sort_values(by=['episode_nb'])
    grouped = results.groupby(['episode_nb'])
    means = grouped.mean()
    stds = grouped.std()
    medians = grouped.median()
    maxs = grouped.max()
    conv_filter = np.full(2 * smoothing + 1, 1 / (2 * smoothing + 1))
    x = means.index[smoothing:(-smoothing) if smoothing else None]
    # print(metric, experiment_ids, np.mean(means[metric]))
    smoothed_means = np.convolve(means[metric], conv_filter, mode='valid')
    smoothed_stds = np.convolve(stds[metric], conv_filter, mode='valid')
    smoothed_medians = np.convolve(medians[metric], conv_filter, mode='valid')
    smoothed_maxs = np.convolve(maxs[metric], conv_filter, mode='valid')
    return SmoothedMetric(x, smoothed_means, smoothed_stds, smoothed_medians, smoothed_maxs)


def plot_metric(ax, db, metric, experiment_ids, color, smoothing=0, reduction_method='mean_std', **plos_kwargs):
    if reduction_method == 'all':
        for experiment_id in experiment_ids:
            results = db.get_dataframe(f"SELECT episode_nb,{metric} FROM results WHERE experiment_id='{experiment_id}'").fillna(np.nan).sort_values(by=['episode_nb'])
            ax.plot(results['episode_nb'], results[metric], color=color)
    elif reduction_method == 'mean_std_5':
        smoothed = get_nlargest_smoothed_metric(db, experiment_ids, metric, smoothing=smoothing, n=5)
        if len(smoothed.x) == len(smoothed.means) == len(smoothed.stds):
            ax.fill_between(
                smoothed.x,
                smoothed.means - smoothed.stds,
                smoothed.means + smoothed.stds,
                color=color, alpha=0.2
            )
        if len(smoothed.x) == len(smoothed.means):
            ax.plot(
                smoothed.x,
                smoothed.means,
                color=color, **plos_kwargs
            )
        return
    elif reduction_method == 'mean_std_2':
        smoothed = get_nlargest_smoothed_metric(db, experiment_ids, metric, smoothing=smoothing, n=2)
        if len(smoothed.x) == len(smoothed.means) == len(smoothed.stds):
            ax.fill_between(
                smoothed.x,
                smoothed.means - smoothed.stds,
                smoothed.means + smoothed.stds,
                color=color, alpha=0.2
            )
        if len(smoothed.x) == len(smoothed.means):
            ax.plot(
                smoothed.x,
                smoothed.means,
                color=color, **plos_kwargs
            )
        return
    smoothed = get_smoothed_metric(db, experiment_ids, metric, smoothing=smoothing)
    if reduction_method == 'mean_std':
        if len(smoothed.x) == len(smoothed.means) == len(smoothed.stds):
            ax.fill_between(
                smoothed.x,
                smoothed.means - smoothed.stds,
                smoothed.means + smoothed.stds,
                color=color, alpha=0.2
            )
        if len(smoothed.x) == len(smoothed.means):
            ax.plot(
                smoothed.x,
                smoothed.means,
                color=color, **plos_kwargs
            )
    elif reduction_method == 'median':
        ax.plot(
            smoothed.x,
            smoothed.medians,
            color=color, **plos_kwargs
        )
    elif reduction_method == 'max':
        ax.plot(
            smoothed.x,
            smoothed.maxs,
            color=color, **plos_kwargs
        )
    elif reduction_method == 'mean_std_all':
        if len(smoothed.x) == len(smoothed.means) == len(smoothed.stds):
            ax.fill_between(
                smoothed.x,
                smoothed.means - smoothed.stds,
                smoothed.means + smoothed.stds,
                color=color, alpha=0.2
            )
        if len(smoothed.x) == len(smoothed.means):
            ax.plot(
                smoothed.x,
                smoothed.means,
                color=color, **plos_kwargs
            )
        for experiment_id in experiment_ids:
            results = db.get_dataframe(f"SELECT episode_nb,{metric} FROM results WHERE experiment_id='{experiment_id}'").fillna(np.nan).sort_values(by=['episode_nb'])
            ax.plot(results['episode_nb'], results[metric], color=color, alpha=0.5)


def get_average_perf(db, metric, collection):
    perfs = {}
    print(collection)
    variation_params = db.get_collection_variation_parameters(collection)
    n_params = len(variation_params.columns)
    if n_params != 1:
        # raise ValueError("this collection has {} != 1 parameters varying ({})".format(n_params, variation_params.columns))
        print("this collection has {} != 1 parameters varying {}".format(n_params, tuple(variation_params.columns)))
    param = variation_params.columns[-1]
    for value in variation_params[param]:
        print(f"{param} = {value}")
        experiment_ids = db.get_experiment_ids(collection=collection, **{param: value})
        df = db.get_dataframe(f'SELECT experiment_id,{metric} FROM results WHERE experiment_id IN {tuple(experiment_ids)}').groupby(['experiment_id']).mean()[metric]
        perfs[value] = df
        print(df)
        print(f"MEAN={df.mean()}")
        print('')
    print('')
    print('')
    print('')
    return perfs


def plot_collection_metric(ax, db, metric, collection, keep=None, smoothing=0, reduction_method='mean_std', ylim=[None, None], ylabel=None, legend_prefix="", above=None):
    log = {}
    variation_params = db.get_collection_variation_parameters(collection)
    n_params = len(variation_params.columns)
    if n_params != 1:
        # raise ValueError("this collection has {} != 1 parameters varying ({})".format(n_params, variation_params.columns))
        print("this collection has {} != 1 parameters varying {}".format(n_params, tuple(variation_params.columns)))
    param = variation_params.columns[-1]
    if keep is not None:
        variation_params = variation_params.loc[keep]
    for value, alpha in zip(variation_params[param], np.linspace(0, 1, len(variation_params[param]))):
        experiment_ids = db.get_experiment_ids(collection=collection, **{param: value})
        if above is not None:
            experiment_ids = [
                experiment_id for experiment_id in experiment_ids
                if db.get_dataframe(f"SELECT {metric} FROM results WHERE experiment_id='{experiment_id}'").mean()[metric] > above
            ]
        log[value] = tuple(db.get_experiment_last_episode_nb(experiment_id) for experiment_id in experiment_ids) # len(experiment_ids)
        color = alpha * stop_color + (1 - alpha) * start_color
        plot_metric(
            ax,
            db,
            metric,
            experiment_ids,
            color=color,
            smoothing=smoothing,
            reduction_method=reduction_method,
            label=f'{legend_prefix}{value:.2f}' if type(value) is float else str(value)
        )
    ax.set_ylim(ylim)
    ax.set_xlabel('#episodes')
    if ylabel is None:
        ax.set_ylabel(metric)
    else:
        ax.set_ylabel(ylabel)
    ax.legend()
    return log


def plot_collection_metric_wrt_param(ax, db, metric, collection, at, smoothing=0, reduction_method='mean_std', ylim=[None, None]):
    variation_params = db.get_collection_variation_parameters(collection)
    n_params = len(variation_params.columns)
    if n_params != 1:
        # raise ValueError("this collection has {} != 1 parameters varying ({})".format(n_params, variation_params.columns))
        print("this collection has {} != 1 parameters varying {}".format(n_params, tuple(variation_params.columns)))
    param = variation_params.columns[-1]
    mean_results = np.full(shape=(len(at), len(variation_params[param])), fill_value=np.nan)
    std_results = np.full(shape=(len(at), len(variation_params[param])), fill_value=np.nan)
    median_results = np.full(shape=(len(at), len(variation_params[param])), fill_value=np.nan)
    max_results = np.full(shape=(len(at), len(variation_params[param])), fill_value=np.nan)
    for i, value in enumerate(variation_params[param]):
        experiment_ids = db.get_experiment_ids(collection=collection, **{param: value})
        smoothed = get_smoothed_metric(db, experiment_ids, metric, smoothing=smoothing)
        for j, episode_nb in enumerate(at):
            where = smoothed.x==episode_nb
            if where.any():
                index = np.argmax(where)
                mean_results[j, i] = smoothed.means[index]
                std_results[j, i] = smoothed.stds[index]
                median_results[j, i] = smoothed.medians[index]
                max_results[j, i] = smoothed.maxs[index]
    for episode_nb, mean, std, median, max, alpha in zip(
            at, mean_results, std_results, median_results, max_results, np.linspace(0, 1, len(at))):
        color = alpha * stop_color + (1 - alpha) * start_color
        # ax.fill_between(variation_params[param], mean - std, mean + std, color=color, alpha=0.2)
        if reduction_method == 'mean_std':
            ax.plot(variation_params[param], mean, color=color, label='@{}'.format(episode_nb))
        elif reduction_method == 'median':
            ax.plot(variation_params[param], median, color=color, label='@{}'.format(episode_nb))
        elif reduction_method == 'max':
            ax.plot(variation_params[param], max, color=color, label='@{}'.format(episode_nb))
    if reduction_method == 'mean_std':
        ax.plot(variation_params[param], np.nanmean(mean_results, axis=0), color='r')
    elif reduction_method == 'median':
        ax.plot(variation_params[param], np.nanmean(median_results, axis=0), color='r')
    elif reduction_method == 'max':
        ax.plot(variation_params[param], np.nanmean(max_results, axis=0), color='r')
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.set_ylim(ylim)
    ax.legend()


def plot_1_variable_param(fig, db, collection, at, keep, smoothing=0, reduction_method='mean_std'):
    metric, ylim = 'success_rate_percent', [-5, 105]
    # metric, ylim = 'delta_distance_to_goal', [0, 2]
    fig.suptitle(collection, fontsize=16)
    ax = fig.add_subplot(221)
    log = plot_collection_metric(ax, db, metric, collection, keep, smoothing=smoothing, reduction_method=reduction_method, ylim=ylim)
    ax = fig.add_subplot(222)
    plot_collection_metric_wrt_param(ax, db, metric, collection, at, smoothing=smoothing, reduction_method=reduction_method, ylim=ylim)
    ax = fig.add_subplot(223)
    plot_collection_metric(ax, db, 'time_to_solve', collection, keep, smoothing=smoothing, reduction_method=reduction_method, ylim=[2, 12])
    ax = fig.add_subplot(224)
    plot_collection_metric_wrt_param(ax, db, 'time_to_solve', collection, at, smoothing=smoothing, reduction_method=reduction_method, ylim=[2, 12])
    # fig.tight_layout(pad=0.50)
    fig.subplots_adjust(left=0.098, bottom=0.105, right=0.988, top=0.924, wspace=0.329, hspace=0.333)
    return log


def plot_db_version_1():
    plt.rc('legend',fontsize=4)

    db = Database('../databases/version_1.db')
    smoothing = 6

    collections_at = (
        (
            'standard_td3_vary_exploration_std',                   # best = 0.4 / 0.5
            [10400, 20000, 40000, 60000, 80000, 98400],
            [0, 2, 5],
        ),
        (
            'standard_td3_vary_exploration_prob',                  # best = 0.85 / 1.0
            [10400, 20000, 40000, 60000, 80000, 98400],
            [0, 3, 6],
        ),
        (
            'standard_td3_vary_exploration_prob_ada_step',         # best = 0.4 / 0.55 (much faster than previous)
            [2400, 4800, 7200, 9600, 12000, 14400],
            [0, 5, 6],
        ),
        (
            'standard_td3_vary_exploration_prob_ada_step_low_nml', # best = 0.1
            [10400, 20000, 40000, 60000, 80000, 98400],
            [0, 4, 6],
        ),
        (
            'standard_td3_no_ada_step_vary_tau',                   # best = equal everywhere
            [10400, 20000, 40000, 60000, 80000, 98400],
            [1, 2, 3],
        ),
        (
            'standard_td3_ada_step_vary_tau',                      # best = equal everywhere
            [2400, 4800, 7200, 9600, 12000, 14400],
            [1, 2, 3],
        ),
        (
            'standard_td3_vary_HER',                                 # best = 2 ?
            [2400, 4800, 7200, 9600, 12000, 14400],
            [0, 1, 3],
        ),
        (
            'trajectory_td3_vary_traj_len',                                 # best = 2 ?
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 3, 5],
        ),
        (
            'standard_td3_vary_timestep', # UPDATE experiments SET collection='standard_td3_vary_timestep' WHERE collection='standard_td3_vary_traj_len';
            [4800, 9600, 14400, 19200, 24000, 28800, 33600, 38400],
            [0, 2, 3],
        ),
        (
            'trajectory_td3_vary_timestep',                          # best =
            [4800, 9600, 14400, 19200, 24000, 28800, 33600, 38400],
            [0, 2, 3],
        ),
        (
            'trajectory_td3_vary_bn_size_3_3',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 3, 5],
        ),
        (
            'trajectory_td3_vary_bn_size_2_2',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 1, 5],
        ),
        (
            'trajectory_td3_vary_bn_size_3_3_prim_critic',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 4, 5],
        ),
        (
            'trajectory_td3_vary_bn_size_2_2_prim_critic',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 4, 5],
        ),
        (
            'trajectory_td3_vary_prim_std',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 3, 5],
        ),
        (
            'trajectory_td3_vary_expl_ratio',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [2, 4, 5],
        ),
        (
            'trajectory_td3_vary_prim_std_3_3',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 3, 5],
        ),
        (
            'trajectory_td3_vary_expl_ratio_3_3',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 3, 6],
        ),
        (
            'trajectory_td3_vary_expl_prob',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 2, 6],
        ),
        (
            'trajectory_td3_vary_expl_prob_3_3',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [1, 3, 5],
        ),
        (
            'trajectory_td3_vary_expl_ratio_3_3_bn_3',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [2, 4],
        ),
        (
            'trajectory_td3_vary_expl_prob_3_3_bn_3',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            [0, 3, 6],
        ),
        (
            'trajectory_td3_overwrite_mvt_df',
            [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200, 20000],
            None,
        ),
    )

    for i, (collection, at, keep) in enumerate(collections_at):
        if i > -1:
            for reduction_method in ['mean_std', 'median', 'max']:
                print('{} {:02d} {}'.format(reduction_method, i, collection))
                fig = plt.figure()
                plot_1_variable_param(fig, db, collection, at, keep, smoothing=smoothing, reduction_method=reduction_method)
                fig.savefig('../plots/{}_{:02d}_{}.png'.format(reduction_method, i, collection), dpi=300)
                plt.close(fig)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='b', smoothing=smoothing, label='TD3')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3_AS')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='r', smoothing=smoothing, label='TD3 + action sequence')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3_AS_BN')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='g', smoothing=smoothing, label='TD3 + action sequence + BN')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3_MP')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='m', smoothing=smoothing, label='TD3 + movement primitives')
    ax.set_xlabel('#episodes')
    ax.set_ylabel('success_rate_percent')
    ax.set_ylim([-5, 105])
    ax.legend()
    fig.savefig('../plots/00_generalization_bn_15.png'.format(i, collection), dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='b', smoothing=smoothing, label='TD3')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3_AS')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='r', smoothing=smoothing, label='TD3 + action sequence')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization_bn_7.TD3_AS_BN')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='g', smoothing=smoothing, label='TD3 + action sequence + BN')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization_bn_7.TD3_MP')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='m', smoothing=smoothing, label='TD3 + movement primitives')
    ax.set_xlabel('#episodes')
    ax.set_ylabel('success_rate_percent')
    ax.set_ylim([-5, 105])
    ax.legend()
    fig.savefig('../plots/00_generalization_bn_7.png'.format(i, collection), dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='b', smoothing=smoothing, label='TD3')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization.TD3_AS')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='r', smoothing=smoothing, label='TD3 + action sequence')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization_bn_3.TD3_AS_BN')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='g', smoothing=smoothing, label='TD3 + action sequence + BN')
    experiment_ids = db.get_experiment_ids(collection='experiment_generalization_bn_3.TD3_MP')
    plot_metric(ax, db, 'success_rate_percent', experiment_ids, color='m', smoothing=smoothing, label='TD3 + movement primitives')
    ax.set_xlabel('#episodes')
    ax.set_ylabel('success_rate_percent')
    ax.set_ylim([-5, 105])
    ax.legend()
    fig.savefig('../plots/00_generalization_bn_3.png'.format(i, collection), dpi=300)



    from collections import defaultdict
    all_exps = db.get_dataframe('''
        SELECT
            experiment_id,
            collection,
            policy_model_arch,
            critic_model_arch,
            policy_primitive_learning_rate,
            policy_movement_learning_rate,
            primitive_exploration_stddev,
            movement_exploration_stddev,
            critic_learning_rate,
            target_smoothing_stddev,
            tau,
            exploration_prob,
            state_size,
            action_size,
            goal_size,
            n_simulations,
            movement_exploration_prob_ratio,
            policy_bottleneck_size,
            policy_default_layer_size,
            critic_default_layer_size,
            environement,
            has_movement_primitive,
            evaluation_goals,
            exploration_goals,
            episode_length_in_sec,
            episode_length_in_it,
            movement_mode,
            discount_factor,
            movement_discount_factor,
            primitive_discount_factor,
            n_actions_in_movement,
            one_action_span_in_sec,
            simulation_timestep,
            updates_per_sample,
            batch_size,
            her_max_replays,
            movement_noise_magnitude_limit,
            primitive_noise_magnitude_limit,
            metabolic_cost_scale,
            buffer_size
        FROM experiments''').fillna("missing")
    grouped_experiments = defaultdict(list)

    for l in all_exps.values:
        grouped_experiments[tuple(l[1:])].append(l[0])

    grouped_perf = {}
    for key, value in grouped_experiments.items():
        grouped_perf[key] = db.get_dataframe('''
            SELECT
                AVG(success_rate_percent)
            FROM
                results
            WHERE
                episode_nb <= 40000
            AND
                experiment_id IN {}'''.format(tuple(value) if len(value) > 1 else '({})'.format(value[0]))).values[0][0]

    for key, value in grouped_experiments.items():
        print(key[0], value, grouped_perf[key])

    grouped_collections = {
        'TD3': [
            'standard_td3_vary_exploration_std',
            'standard_td3_vary_exploration_prob',
            'standard_td3_vary_exploration_prob_ada_step',
            'standard_td3_vary_exploration_prob_ada_step_low_nml',
            'standard_td3_no_ada_step_vary_tau',
            'standard_td3_ada_step_vary_tau',
            'standard_td3_vary_HER',
            'standard_td3_vary_timestep',
        ],
        'TD3 + action sequence': [
            'trajectory_td3_vary_traj_len',
            'trajectory_td3_vary_timestep',
            'trajectory_td3_vary_bn_size_3_3',
            'trajectory_td3_vary_bn_size_2_2',
        ],
        'TD3 + movement primitive': [
            'trajectory_td3_vary_bn_size_3_3_prim_critic',
            'trajectory_td3_vary_bn_size_2_2_prim_critic',
            'trajectory_td3_vary_prim_std',
            'trajectory_td3_vary_expl_ratio',
            'trajectory_td3_vary_prim_std_3_3',
            'trajectory_td3_vary_expl_ratio_3_3',
            'trajectory_td3_vary_expl_prob',
            'trajectory_td3_vary_expl_prob_3_3',
            'trajectory_td3_vary_expl_ratio_3_3_bn_3',
            'trajectory_td3_vary_expl_prob_3_3_bn_3',
        ],
    }

    bests = {}
    for experiment_type, collections in grouped_collections.items():
        best = -1
        for key in grouped_experiments:
            if key[0] in collections:
                if grouped_perf[key] > best:
                    best = grouped_perf[key]
                    experiment_ids = grouped_experiments[key]
        bests[experiment_type] = experiment_ids

    print(bests)

    plt.rc('legend',fontsize=14)
    colors = ('b', 'r', 'g')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (key, experiment_ids), color in zip(bests.items(), colors):
        plot_metric(ax, db, 'success_rate_percent', experiment_ids, color, smoothing=smoothing, label=key)
    ax.legend()
    ax.set_xlabel('#episodes')
    ax.set_ylabel('success_rate_percent')
    ax.set_ylim([-5, 105])
    fig.savefig('../plots/00_bests_by_cat.png', dpi=300)

    sorted_keys = list(sorted(grouped_perf, key=grouped_perf.get, reverse=True))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_metric(ax, db, 'success_rate_percent', grouped_experiments[sorted_keys[0]], 'b', smoothing=smoothing)
    print(sorted_keys[0])
    plot_metric(ax, db, 'success_rate_percent', grouped_experiments[sorted_keys[1]], 'r', smoothing=smoothing)
    print(sorted_keys[1])
    plot_metric(ax, db, 'success_rate_percent', grouped_experiments[sorted_keys[2]], 'g', smoothing=smoothing)
    print(sorted_keys[2])
    plot_metric(ax, db, 'success_rate_percent', grouped_experiments[sorted_keys[3]], 'c', smoothing=smoothing)
    print(sorted_keys[3])
    plot_metric(ax, db, 'success_rate_percent', grouped_experiments[sorted_keys[4]], 'm', smoothing=smoothing)
    print(sorted_keys[4])
    plot_metric(ax, db, 'success_rate_percent', grouped_experiments[sorted_keys[5]], 'y', smoothing=smoothing)
    print(sorted_keys[5])
    # plot_metric(ax, db, 'success_rate_percent', grouped_experiments[sorted_keys[3]], 'k', smoothing=smoothing)
    fig.savefig('../plots/00_bests.png', dpi=300)


def plot_db_thesis_experiments():
    plt.rc('legend',fontsize=4)

    db = Database('../databases/thesis_experiments.db')
    smoothing = 6

    collections_at = (
        (
            "standard_vary_exploration_std",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "standard_vary_exploration_prob",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "bell_vary_exploration_std",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "bell_vary_exploration_prob",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "trajectory_vary_sequence_length",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "trajectory_vary_exploration_std",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "trajectory_vary_exploration_prob",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "trajectory_bn_vary_bn",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "hrl_vary_exploration_ratio",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "hrl_vary_exploration_prob",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "standard_vary_exploration_prob_ada_small",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "bell_vary_exploration_prob_ada_small",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
        (
            "trajectory_vary_exploration_prob_ada_small",
            [10400, 20000, 40000, 60000, 80000, 98400],
            None,
        ),
    )

    for i, (collection, at, keep) in enumerate(collections_at):
        if i > 10:
            for reduction_method in ['mean_std', 'median', 'max']:
                print('{}_{:02d}_{}'.format(reduction_method, i, collection))
                fig = plt.figure()
                plot_1_variable_param(fig, db, collection, at, keep, smoothing=smoothing, reduction_method=reduction_method)
                fig.savefig('../plots_2/{}_{:02d}_{}.png'.format(reduction_method, i, collection), dpi=300)
                plt.close(fig)


if __name__ == '__main__':
    # plot_db_version_1()
    plot_db_thesis_experiments()

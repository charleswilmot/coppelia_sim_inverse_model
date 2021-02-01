import tikzplotlib
from tensorboard.data.experimental import ExperimentFromDev
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pickle


def group_by_repetition(scalars):
    runs = scalars.run.unique()
    # each run has name job*_A               --> no repetition
    #                or job*_Arepetition.*C  --> repetition id indicated by second star
    # job[0-9]+_([a-zA-Z0-9\._\/]*)repetition\.([0-9]+)([a-zA-Z0-9\._\/]*)
    # job[0-9]+_([a-zA-Z0-9\._\/]*)
    groups = defaultdict(list)
    for run in runs:
        match_repetition = re.match("[0-9\-\/]*job[0-9]+_([a-zA-Z0-9\._\/\-]*)repetition\.([0-9]+)([a-zA-Z0-9\._\/]*)", run)
        match_no_repetition = re.match("[0-9\-\/]*job[0-9]+_([a-zA-Z0-9\._\/]*)", run)
        if match_repetition:
            A = match_repetition.group(1)
            C = match_repetition.group(3)
            groups[(A, C)].append(run)
        elif match_no_repetition:
            A = match_no_repetition.group(1)
            groups[A].append(run)
        else:
            print("job name could not be match with a regex: {} , skipping".format(run))
    print("Found {} groups:".format(len(groups)))
    for x in groups:
        if isinstance(x, tuple):
            print("job*_{}repetition.*{}".format(x[0], x[1]))
        elif isinstance(x, str):
            print("job*_{}".format(x))
    print("\n")
    renamed_groups = defaultdict(list)
    try:
        with open("../tmp/default_short_names.pkl", 'rb') as f:
            default_names = pickle.load(f)
    except FileNotFoundError as e:
        default_names = {}
    for x in groups:
        if x in default_names:
            def_name = default_names[x]
            suffix = "default = {}".format(def_name)
        else:
            suffix = ""
        if isinstance(x, tuple):
            name = input("Please give a short name to job*_{}repetition.*{}\n{}\n".format(x[0], x[1], suffix)) or def_name
        elif isinstance(x, str):
            name = input("Please give a short name to job*_{}\n{}\n".format(x, suffix)) or def_name
        if name != "del":
            renamed_groups[name] += groups[x]
        default_names[x] = name
    with open("../tmp/default_short_names.pkl", 'wb') as f:
        pickle.dump(default_names, f)
    return renamed_groups


def get_mean_std(data):
    all_lengths = sorted([d.step.values[-1] for d in data])
    all_starts = sorted([d.step.values[0] for d in data])
    max_start = max(all_starts)
    min_length = min(all_lengths)
    max_length = max(all_lengths)
    std_limit = all_lengths[-2]
    x = np.arange(max_start, max_length)
    data = [np.interp(x[x <= d.step.values[-1]], d.step, d["value"]) for d in data]
    sum_arr = np.zeros(max_length - max_start)
    count_arr = np.zeros(max_length - max_start, dtype=np.int32)
    for d in data:
        sum_arr[:len(d)] += d
        count_arr[:len(d)] += 1
    mean = sum_arr / count_arr

    sum_arr = np.zeros(max_length - max_start)
    count_arr = np.zeros(max_length - max_start, dtype=np.int32)
    for d in data:
        sum_arr[:len(d)] += (d - mean[:len(d)]) ** 2
        count_arr[:len(d)] += 1
    std = np.sqrt(sum_arr / count_arr)
    return x, mean, std


def plot_by_tag(fig, scalars, groups, tag, ylim=None):
    ax = fig.add_subplot(111)
    for name, runs in groups.items(): # for each group
        data = [scalars[scalars.run.eq(run) & scalars.tag.eq(tag)] for run in runs]
        x, mean, std = get_mean_std(data)
        line, = ax.plot(x, mean, label=name)
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.1)
    scalar_name = tag.split("/")[-1].replace('_', ' ')
    n_repetitions = set(len(runs) for runs in groups.values())
    if len(n_repetitions) == 1:
        suffix = "  ({} repetitions)".format(n_repetitions.pop())
    else:
        suffix = ""
    fig.suptitle(scalar_name + suffix)
    ax.legend()
    ax.set_xlabel("episodes")
    ax.set_ylabel(scalar_name)
    if ylim is not None:
        ax.set_ylim(ylim)


def aggregate_runs(experiment_id, path):
    exp = ExperimentFromDev(experiment_id)
    scalars = exp.get_scalars()
    groups = group_by_repetition(scalars)
    available_groups_string = ""
    for i, key in enumerate(groups):
        available_groups_string += "{: 2d} \t {}\n".format(i, key)
    fig = plt.figure(dpi=300)
    done = False
    while not done:
        which = list(map(int, input("Which groups should be plotted? available are:\n" + available_groups_string).split(',')))
        groups_to_plot = {key: value for i, (key, value) in enumerate(groups.items()) if i in which}
        for tag, ylim in [
                ("evaluation_success_rate_percent_wrt_ep", (0, 105)),
                ("evaluation_success_rate_percent_wrt_tr", (0, 105)),
                ("exploration_success_rate_percent_wrt_ep", (0, 105)),
                ("exploration_success_rate_percent_wrt_tr", (0, 105)),
                ("evaluation_delta_distance_to_goal_wrt_ep", (0, 2.0)),
                ("evaluation_delta_distance_to_goal_wrt_tr", (0, 2.0)),
                ("exploration_delta_distance_to_goal_wrt_ep", (0, 2.0)),
                ("exploration_delta_distance_to_goal_wrt_tr", (0, 2.0)),
                ("evaluation_time_to_solve_wrt_ep", (0, 25)),
                ("evaluation_time_to_solve_wrt_tr", (0, 25)),
                ]:
            plot_by_tag(fig, scalars, groups_to_plot, "collection/{}".format(tag), ylim=ylim)
            fig.savefig(path + "/{}_{}_{}.png".format(tag, "_".join(map(str, which)), experiment_id))
            # tikzplotlib.save(path + "/{}_{}_{}.tex".format(tag, "_".join(map(str, which)), experiment_id))
            fig.clf(fig)
        done = input("make an other plot? (yes/no)") == "no"
    plt.close(fig)


if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]
    aggregate_runs(experiment_id, '/tmp')

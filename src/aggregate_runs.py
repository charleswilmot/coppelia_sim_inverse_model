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
        match_repetition = re.match("job[0-9]+_([a-zA-Z0-9\._\/]*)repetition\.([0-9]+)([a-zA-Z0-9\._\/]*)", run)
        match_no_repetition = re.match("job[0-9]+_([a-zA-Z0-9\._\/]*)", run)
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
        elif isinstance(x, string):
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
        elif isinstance(x, string):
            name = input("Please give a short name to job*_{}\n{}\n".format(x, suffix)) or def_name
        if name != "del":
            renamed_groups[name] += groups[x]
        default_names[x] = name
    with open("../tmp/default_short_names.pkl", 'wb') as f:
        pickle.dump(default_names, f)
    return renamed_groups


def plot_by_tag(fig, scalars, groups, tag):
    ax = fig.add_subplot(111)
    for name, runs in groups.items(): # for each group
        data = [scalars[scalars.run.eq(run) & scalars.tag.eq(tag)] for run in runs]
        x = data[0].step
        mean = np.mean([d["value"] for d in data], axis=0)
        line, = ax.plot(x, mean, label=name)
        if len(runs) > 1:
            std = np.std([d["value"] for d in data], axis=0)
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


def aggregate_runs(experiment_id, path):
    exp = ExperimentFromDev(experiment_id)
    scalars = exp.get_scalars()
    groups = group_by_repetition(scalars)
    fig = plt.figure(dpi=300)
    for tag in [
            "evaluation_success_rate_percent",
            "exploration_success_rate_percent",
            "evaluation_delta_distance_to_goal",
            "exploration_delta_distance_to_goal",
            ]:
        plot_by_tag(fig, scalars, groups, "collection/{}".format(tag))
        fig.savefig(path + "/{}.png".format(tag))
        fig.clf(fig)
    plt.close(fig)


if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]
    aggregate_runs(experiment_id, '/tmp')

from utils import load_appendable_array_file
import numpy as np
import matplotlib.pyplot as plt


path = '../experiments/2021-02-05/10-31-34/job01_a.policy_movement_learning_rate.8e-05__a.policy_model_arch.movement_primitive_3_3__e.n_episodes.40000__e.repetition.1__p.episode_length_in_sec.60__p.movement_mode.full_raw__p.movement_span_in_sec.1.2__p.overwrite_movement_discount_fact/visualization_data/evaluation_goals.dat'
data = load_appendable_array_file(path)
print(data.shape)


def get_success(data):
    return (data['goals'] == data['current_goals']).all(axis=-1).any(axis=-1)

def get_trivial(data):
    return (data['goals'][..., 0, :] == data['current_goals'][..., 0, :]).all(axis=-1)

def get_success_not_trivial(data):
    return np.logical_and(get_success(data), np.logical_not(get_trivial(data)))

def get_iteration_first_success(data):
    trivial = get_trivial(data)
    ret = np.argmax((data['goals'] == data['current_goals']).all(axis=-1), axis=-1)
    return np.where(np.logical_and(ret == 0, np.logical_not(trivial)), data.shape[-1], ret)

def get_goal_size(data):
    return data['goals'].shape[-1]

def get_goal_set(data):
    goal_size = get_goal_size(data)
    return set(tuple(x) for x in data['goals'].reshape((-1, goal_size)))

def get_goals_per_episode(data):
    return data['goals'][:, 0]

def get_per_goal(data, func, data_sample_rate=1):
    goal_set = get_goal_set(data)
    goals_per_episode = get_goals_per_episode(data)
    measure = func(data)
    ret = {}
    for goal in goal_set:
        indices = (goals_per_episode == goal).all(axis=-1).nonzero()[0]
        ret[goal] = (indices * data_sample_rate, measure[indices])
    return ret

def get_switch_set(data):
    switch_size = get_goal_size(data)
    return set(tuple(x) for x in np.logical_xor(data['goals'][:, 0], data['current_goals'][:, 0]).reshape((-1, switch_size)))

def get_switches_per_episode(data):
    return np.logical_xor(data['goals'][:, 0], data['current_goals'][:, 0])

def get_per_switch(data, func, data_sample_rate=1):
    switch_set = get_switch_set(data)
    switches_per_episode = get_switches_per_episode(data)
    measure = func(data)
    ret = {}
    for switch in switch_set:
        indices = (switches_per_episode == switch).all(axis=-1).nonzero()[0]
        ret[switch] = (indices * data_sample_rate, measure[indices])
    return ret

def profile(x, width=2000):
    return np.exp(- x ** 2 / width ** 2)

def interpolate(per_goal, width=2000, N=200):
    n_episodes = max([x[0][-1] for x in per_goal.values()])
    full_per_goal = {}
    X = np.linspace(0, n_episodes, N)
    mean_values = np.zeros(N)
    confidences = np.zeros(N)
    for goal, (indices, values) in per_goal.items():
        for i in range(N):
            weights = profile(X[i] - indices, width=width)
            mean_values[i] = np.average(values, weights=weights)
            confidences[i] = np.sum(weights)
        full_per_goal[goal] = (np.copy(mean_values), np.copy(confidences))
    return X, full_per_goal


# print(get_success(data))
# print("\n###\n")
# print(get_trivial(data))
# print("\n###\n")
# print(get_iteration_first_success(data))
# print("\n###\n")
# print(get_goal_size(data))
# print("\n###\n")
# print(get_goal_set(data))
#
# print("\n###\n")
# print("\n###\n")
# print("\n###\n")
#
# print(get_per_goal(data, get_success))
# print("\n###\n")
# print(get_per_goal(data, get_trivial))
# print("\n###\n")
# print(get_per_goal(data, get_success_not_trivial))
# print("\n###\n")
# print(per_goal_iteration_first_success)
# print("\n###\n")



def get_color(goal):
    if goal[1]:
        return [1, 0, 0] # red
    if goal[3]:
        return [0, 1, 0] # green
    if goal[0]:
        return [0, 0, 1] # blue
    if goal[2]:
        return [1, 0, 1] # pink
    return [0, 0, 0]     # black


# # per_goal_iteration_first_success = get_per_goal(data, get_iteration_first_success, data_sample_rate=20)
# per_goal_iteration_first_success = get_per_goal(data, get_success_not_trivial, data_sample_rate=20)
# # per_goal_iteration_first_success = get_per_switch(data, get_iteration_first_success, data_sample_rate=20)
# # per_goal_iteration_first_success = get_per_switch(data, get_success_not_trivial, data_sample_rate=20)
# X, full_per_goal_iteration_first_success = interpolate(per_goal_iteration_first_success)
#
# for goal, data in full_per_goal_iteration_first_success.items():
#     plt.plot(X, data[0], label=str(goal), color=get_color(goal))
# plt.show()


def confidence_plot(ax, x, y, conf, color=[0, 0, 1]):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(conf.min(), np.percentile(conf, 95), clip=True)
    colorlist = np.concatenate([
        np.repeat(np.array(color)[np.newaxis], 256, axis=0),
        np.linspace(0, 1, 256)[:, np.newaxis]
    ], axis=1)
    cmap = ListedColormap(colorlist)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(conf)
    lc.set_linewidth(1)
    ax.set_xlim([x.min(), x.max()])
    delta_lim = (y.max() - y.min()) * 0.05
    ax.set_ylim([y.min() - delta_lim, y.max() + delta_lim])
    return ax.add_collection(lc)


def plot(ax_per_goal_iteration_first_success,
        ax_per_goal_success_not_trivial,
        ax_per_switch_iteration_first_success,
        ax_per_switch_success_not_trivial,
        data, data_sample_rate=1):
    per_goal_iteration_first_success = get_per_goal(data, get_iteration_first_success, data_sample_rate=data_sample_rate)
    per_goal_success_not_trivial = get_per_goal(data, get_success_not_trivial, data_sample_rate=data_sample_rate)
    per_switch_iteration_first_success = get_per_switch(data, get_iteration_first_success, data_sample_rate=data_sample_rate)
    per_switch_success_not_trivial = get_per_switch(data, get_success_not_trivial, data_sample_rate=data_sample_rate)

    X, full_per_goal_iteration_first_success = interpolate(per_goal_iteration_first_success)
    X, full_per_goal_success_not_trivial = interpolate(per_goal_success_not_trivial)
    X, full_per_switch_iteration_first_success = interpolate(per_switch_iteration_first_success)
    X, full_per_switch_success_not_trivial = interpolate(per_switch_success_not_trivial)

    for goal, (mean, confidence) in full_per_goal_iteration_first_success.items():
        confidence_plot(ax_per_goal_iteration_first_success, X, mean, confidence, color=get_color(goal))
    ax_per_goal_iteration_first_success.set_ylim([0, None])
    ax_per_goal_iteration_first_success.set_title("iteration first success (for each goal)")

    for goal, (mean, confidence) in full_per_goal_success_not_trivial.items():
        confidence_plot(ax_per_goal_success_not_trivial, X, mean, confidence, color=get_color(goal))
    ax_per_goal_success_not_trivial.set_ylim([-0.05, 1.05])
    ax_per_goal_success_not_trivial.set_title("success rate (for each goal)")

    for goal, (mean, confidence) in full_per_switch_iteration_first_success.items():
        confidence_plot(ax_per_switch_iteration_first_success, X, mean, confidence, color=get_color(goal))
    ax_per_switch_iteration_first_success.set_ylim([0, None])
    ax_per_switch_iteration_first_success.set_title("iteration first success (for each switch)")

    for goal, (mean, confidence) in full_per_switch_success_not_trivial.items():
        confidence_plot(ax_per_switch_success_not_trivial, X, mean, confidence, color=get_color(goal))
    ax_per_switch_success_not_trivial.set_ylim([-0.05, 1.05])
    ax_per_switch_success_not_trivial.set_title("success rate (for each switch)")


def plot_one_set(fig, data, title, data_sample_rate=1):
    ax_per_goal_iteration_first_success = fig.add_subplot(221)
    ax_per_goal_success_not_trivial = fig.add_subplot(222)
    ax_per_switch_iteration_first_success = fig.add_subplot(223)
    ax_per_switch_success_not_trivial = fig.add_subplot(224)

    plot(
        ax_per_goal_iteration_first_success,
        ax_per_goal_success_not_trivial,
        ax_per_switch_iteration_first_success,
        ax_per_switch_success_not_trivial,
        data,
        data_sample_rate=data_sample_rate
    )
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.2, hspace=0.4)


def plot_two_sets(fig, data_training, data_eval, training_sample_rate=1, eval_sample_rate=20):
    ax_per_goal_iteration_first_success = fig.add_subplot(241)
    ax_per_goal_success_not_trivial = fig.add_subplot(242)
    ax_per_switch_iteration_first_success = fig.add_subplot(243)
    ax_per_switch_success_not_trivial = fig.add_subplot(244)

    plot(
        ax_per_goal_iteration_first_success,
        ax_per_goal_success_not_trivial,
        ax_per_switch_iteration_first_success,
        ax_per_switch_success_not_trivial,
        data_training,
        data_sample_rate=training_sample_rate
    )

    ax_per_goal_iteration_first_success = fig.add_subplot(245)
    ax_per_goal_success_not_trivial = fig.add_subplot(246)
    ax_per_switch_iteration_first_success = fig.add_subplot(247)
    ax_per_switch_success_not_trivial = fig.add_subplot(248)

    plot(
        ax_per_goal_iteration_first_success,
        ax_per_goal_success_not_trivial,
        ax_per_switch_iteration_first_success,
        ax_per_switch_success_not_trivial,
        data_eval,
        data_sample_rate=eval_sample_rate
    )
    fig.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.figtext(0.05, 0.17, 'Evaluation', rotation=90, fontsize=20, horizontalalignment='center')
    plt.figtext(0.05, 0.6, 'Training', rotation=90, fontsize=20, horizontalalignment='center')



if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.collections import LineCollection

    path = sys.argv[1]

    path_eval = path + '/evaluation_goals.dat'
    data_eval = load_appendable_array_file(path_eval)
    path_training = path + '/training_goals.dat'
    data_training = load_appendable_array_file(path_training)



    fig = plt.figure(figsize=(8, 6), dpi=400)
    plot_one_set(fig, data_eval, 'Evaluation', data_sample_rate=20)
    fig.savefig(path + '/evaluation_goals_success.png')
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=400)
    plot_one_set(fig, data_training, 'Training', data_sample_rate=1)
    fig.savefig(path + '/training_goals_success.png')
    plt.close(fig)

    fig = plt.figure(figsize=(16, 6), dpi=400)
    plot_two_sets(fig, data_training, data_eval)
    fig.savefig(path + '/goals_success.png')
    plt.close(fig)

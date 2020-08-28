import tensorflow as tf
from tensorflow import keras
import numpy as np
from custom_layers import custom_objects


def divide_no_nan(a, b, default=0.0):
    return np.divide(a, b, out=np.full_like(a, fill_value=default), where=b!=0)


def model_copy(model, fake_inp):
    clone = keras.models.clone_model(model)
    fake_out = model(fake_inp)
    fake_out = clone(fake_inp)
    for model_var, clone_var in zip(model.variables, clone.variables):
        clone_var.assign(model_var)
    return clone


class Agent(object):
    def __init__(self,
            policy_learning_rate, policy_model_arch,
            critic_learning_rate, critic_model_arch,
            forward_learning_rate, forward_model_arch,
            exploration, target_smoothing_stddev, tau,
            state_size, action_size, goal_size, n_simulations):
        #   POLICY
        self.policy_learning_rate = policy_learning_rate
        self.policy_model = keras.models.model_from_yaml(
            policy_model_arch.pretty(resolve=True),
            custom_objects=custom_objects
        )
        fake_inp = np.zeros(
            shape=(1, state_size + goal_size),
            dtype=np.float32
        )
        self.target_policy_model = model_copy(self.policy_model, fake_inp)
        self.policy_optimizer = keras.optimizers.Adam(self.policy_learning_rate)
        #   CRITIC
        self.critic_learning_rate = critic_learning_rate
        self.critic_model_0 = keras.models.model_from_yaml(
            critic_model_arch.pretty(resolve=True),
            custom_objects=custom_objects
        )
        self.critic_model_1 = keras.models.clone_model(self.critic_model_0)
        fake_inp = np.zeros(
            shape=(1, state_size + goal_size + eval(str(action_size))),
            dtype=np.float32
        )
        self.target_critic_model_0 = model_copy(self.critic_model_0, fake_inp)
        self.target_critic_model_1 = model_copy(self.critic_model_1, fake_inp)
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)
        #   FORWARD
        self.forward_learning_rate = forward_learning_rate
        self.forward_model = keras.models.model_from_yaml(
            forward_model_arch.pretty(resolve=True),
            custom_objects=custom_objects
        )
        self.forward_optimizer = keras.optimizers.Adam(self.forward_learning_rate)
        #   EXPLORATION NOISE
        self.exploration_params = exploration
        self.exploration_stddev = tf.Variable(exploration.stddev, dtype=tf.float32)
        self.exploration_n = exploration.n
        self.success_rate = None
        self.autotune_scale = exploration.autotune_scale
        self.success_rate_estimator_speed = exploration.success_rate_estimator_speed
        self.n_simulations = n_simulations
        if self.n_simulations != 1:
            self.stddev_coefs_step = self.autotune_scale ** -(2 / (self.n_simulations - 1))
            self.histogram_step = self.stddev_coefs_step ** 2
            self.stddev_coefs = self.stddev_coefs_step ** np.arange(
                -(self.n_simulations - 1) / 2,
                1 + (self.n_simulations - 1) / 2,
                1
            )
            self.bins = self.histogram_step ** np.arange(
                np.floor(np.log(0.0001) / np.log(self.histogram_step)),
                np.ceil(np.log(2) / np.log(self.histogram_step))
            )
            self.mean_reward_sum = np.zeros(len(self.bins) + 1)
            self.mean_reward_count = np.zeros(len(self.bins) + 1)
        #   TD3
        self.target_smoothing_stddev = target_smoothing_stddev
        self.tau = tau

    def save_weights(self, path):
        self.policy_model.save_weights(path + "/policy_model")
        self.critic_model_0.save_weights(path + "/critic_model_0")
        self.critic_model_1.save_weights(path + "/critic_model_1")
        self.target_critic_model_0.save_weights(path + "/target_critic_model_0")
        self.target_critic_model_1.save_weights(path + "/target_critic_model_1")

    def load_weights(self, path):
        self.policy_model.load_weights(path + "/policy_model")
        self.critic_model_0.load_weights(path + "/critic_model_0")
        self.critic_model_1.load_weights(path + "/critic_model_1")
        self.target_critic_model_0.load_weights(path + "/target_critic_model_0")
        self.target_critic_model_1.load_weights(path + "/target_critic_model_1")

    @tf.function
    def get_actions(self, states, goals, exploration=False, target=False):
        inps = tf.concat([states, tf.cast(goals, tf.float32)], axis=-1)
        if target or exploration:
            if target:
                stddev = self.target_smoothing_stddev
                pure_actions = self.target_policy_model(inps)
            else:
                stddev = self.exploration_stddev
                pure_actions = self.policy_model(inps)
            noises = tf.random.truncated_normal(
                shape=tf.shape(pure_actions),
                stddev=stddev,
            )
            noisy_actions = tf.clip_by_value(
                pure_actions + noises,
                clip_value_min=-1,
                clip_value_max=1
            )
            noises = noisy_actions - pure_actions
            return pure_actions, noisy_actions, noises
        else:
            return self.policy_model(inps)

    def register_total_reward(self, rewards):
        stddevs = self.stddev_coefs * self.exploration_stddev
        current_bins = np.digitize(stddevs, self.bins)
        c = self.success_rate_estimator_speed
        print('current_bins', current_bins)
        print('rewards', rewards)
        for bin, reward in zip(current_bins, rewards):
            self.mean_reward_sum[bin] = reward + (1 - c) * self.mean_reward_sum[bin]
            self.mean_reward_count[bin] = 1 + (1 - c) * self.mean_reward_count[bin]
        mean_reward = divide_no_nan(self.mean_reward_sum, self.mean_reward_count, default=-np.inf)
        index = np.argmax(mean_reward)
        print('mean_reward', mean_reward)
        if index == 0:
            best_std = self.bins[0]
        elif index == len(self.bins):
            best_std = self.bins[-1]
        else:
            best_std = 0.5 * (self.bins[index - 1] + self.bins[index])
        best_std = c * min(best_std, 1.0) + (1 - c) * self.exploration_stddev.numpy()
        self.exploration_stddev.assign(best_std)

    @tf.function
    def get_predictions(self, states, actions):
        inps = tf.concat([states, actions], axis=-1)
        return self.forward_model(inps)

    @tf.function
    def get_return_estimates(self, states, actions, goals, target=False):
        inps = tf.concat([states, actions, tf.cast(goals, tf.float32)], axis=-1)
        if target:
            target_0 = self.target_critic_model_0(inps)
            target_1 = self.target_critic_model_1(inps)
            return tf.minimum(target_0, target_1)
        else:
            return (self.critic_model_0(inps) + self.critic_model_1(inps)) / 2

    @tf.function
    def train_critic(self, states, actions, goals, targets):
        with tf.GradientTape() as tape:
            estimates = self.get_return_estimates(states, actions, goals)
            loss_critic = keras.losses.Huber()(estimates, tf.stop_gradient(targets))
            vars = self.critic_model_0.variables + self.critic_model_1.variables
            grads = tape.gradient(loss_critic, vars)
            self.critic_optimizer.apply_gradients(zip(grads, vars))
        return loss_critic

    @tf.function
    def train_policy(self, states, goals):
        with tf.GradientTape() as tape:
            actions = self.get_actions(states, goals, exploration=False)
            estimates = self.get_return_estimates(states, actions, goals)
            loss = - tf.reduce_sum(estimates)
            vars = self.policy_model.variables
            grads = tape.gradient(loss, vars)
            self.policy_optimizer.apply_gradients(zip(grads, vars))
        return loss

    @tf.function
    def train_forward(self, states, actions, targets):
        with tf.GradientTape() as tape:
            predictions = self.get_predictions(states, actions)
            losses = keras.losses.MSE(predictions, targets)
            loss = tf.reduce_sum(tf.reduce_mean(losses, axis=-1))
            vars = self.forward_model.variables
            grads = tape.gradient(loss, vars)
            self.forward_optimizer.apply_gradients(zip(grads, vars))
        return loss

    @tf.function
    def update_targets(self):
        model_target_pairs = [
            (self.critic_model_0, self.target_critic_model_0),
            (self.critic_model_1, self.target_critic_model_1),
            (self.policy_model, self.target_policy_model),
        ]
        for model, target in model_target_pairs:
            for model_var, target_var in zip(model.variables, target.variables):
                target_var.assign(
                    (1 - self.tau) * target_var +
                    self.tau * model_var
                )

    @tf.function
    def train(self, states, actions, goals, critic_target, forward_target,
            policy=True, critic=True, forward=True):
        losses = {}
        if critic:
            critic_loss = self.train_critic(
                states, actions, goals, critic_target)
            losses["critic"] = critic_loss
        if forward:
            forward_loss = self.train_forward(states, actions, forward_target)
            losses["forward"] = forward_loss
        if policy:
            policy_loss = self.train_policy(states, goals)
            losses["policy"] = policy_loss
        self.update_targets()
        return losses

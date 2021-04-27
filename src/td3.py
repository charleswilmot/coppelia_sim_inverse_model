import numpy as np
import tensorflow as tf
from tensorflow import keras


def model_copy(model, fake_inp):
    clone = keras.models.clone_model(model)
    fake_out = model(fake_inp)
    fake_out = clone(fake_inp)
    for model_var, clone_var in zip(model.variables, clone.variables):
        clone_var.assign(model_var)
    return clone


class TD3(object):
    def __init__(self, policy_learning_rate, policy_model,
                       critic_learning_rate, critic_model,
                       exploration_stddev, target_smoothing_stddev, tau,
                       policy_state_size, critic_state_size, action_size,
                       n_simulations):
        self.exploration_stddev = exploration_stddev
        self.target_smoothing_stddev = target_smoothing_stddev
        self.tau = tau
        self.action_size = action_size
        self.policy_state_size = policy_state_size
        self.critic_state_size = critic_state_size
        #   POLICY
        self.policy_learning_rate = policy_learning_rate
        self.policy_model = policy_model
        fake_inp = np.zeros(
            shape=(n_simulations, self.policy_state_size),
            dtype=np.float32
        )
        self.target_policy_model = model_copy(self.policy_model, fake_inp)
        self.policy_optimizer = keras.optimizers.Adam(self.policy_learning_rate)
        #   CRITIC
        self.critic_learning_rate = critic_learning_rate
        self.critic_model_0 = critic_model
        self.critic_model_1 = keras.models.clone_model(self.critic_model_0)
        fake_inp = np.zeros(
            shape=(n_simulations, self.critic_state_size + self.action_size),
            dtype=np.float32
        )
        self.target_critic_model_0 = model_copy(self.critic_model_0, fake_inp)
        self.target_critic_model_1 = model_copy(self.critic_model_1, fake_inp)
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)

    def save_weights(self, path):
        self.policy_model.save_weights(path + "/policy_model")
        self.target_policy_model.save_weights(path + "/target_policy_model")
        self.critic_model_0.save_weights(path + "/critic_model_0")
        self.critic_model_1.save_weights(path + "/critic_model_1")
        self.target_critic_model_0.save_weights(path + "/target_critic_model_0")
        self.target_critic_model_1.save_weights(path + "/target_critic_model_1")

    def load_weights(self, path,
            policy_model=True, target_policy_model=True, critic_model_0=True,
            critic_model_1=True, target_critic_model_0=True, target_critic_model_1=True):
        if policy_model:
            self.policy_model.load_weights(path + "/policy_model")
        if target_policy_model:
            self.target_policy_model.load_weights(path + "/policy_model")
        if critic_model_0:
            self.critic_model_0.load_weights(path + "/critic_model_0")
        if critic_model_1:
            self.critic_model_1.load_weights(path + "/critic_model_1")
        if target_critic_model_0:
            self.target_critic_model_0.load_weights(path + "/target_critic_model_0")
        if target_critic_model_1:
            self.target_critic_model_1.load_weights(path + "/target_critic_model_1")

    @tf.function
    def get_actions(self, policy_states, target=False, explore=False):
        new_shape = tf.concat([tf.shape(policy_states)[:-1], [-1], [self.action_size]], axis=0)
        if target:
            pure_actions = self.target_policy_model(policy_states, training=False)
            pure_actions = tf.reshape(pure_actions, new_shape)
            noises = tf.random.truncated_normal(
                shape=tf.shape(pure_actions),
                stddev=self.target_smoothing_stddev,
            )
            noisy_actions = pure_actions + noises
        elif explore is False:
            pure_actions = self.policy_model(policy_states, training=False)
            pure_actions = tf.reshape(pure_actions, new_shape)
            noisy_actions = pure_actions
            noises = tf.zeros_like(pure_actions)
        else:
            pure_actions = self.policy_model(policy_states, training=True)
            pure_actions = tf.reshape(pure_actions, new_shape)
            noises = tf.random.truncated_normal(
                shape=tf.shape(pure_actions),
                stddev=self.exploration_stddev,
            ) * tf.cast(tf.reshape(explore, (-1, 1, 1)), tf.float32)
            noisy_actions_non_clipped = pure_actions + noises
            noisy_actions = tf.clip_by_value(noisy_actions_non_clipped, -1, 1)
            noises = noisy_actions - pure_actions
        return pure_actions, noisy_actions, noises

    @tf.function
    def get_return_estimates(self, critic_states, actions, target=False, mode='mean'):
        inps = tf.concat([critic_states, actions], axis=-1)
        if target:
            target_0 = self.target_critic_model_0(inps)
            target_1 = self.target_critic_model_1(inps)
            return tf.minimum(target_0, target_1)
        else:
            if mode == 'mean':
                return (self.critic_model_0(inps) + self.critic_model_1(inps)) / 2
            elif mode == 'both':
                return self.critic_model_0(inps), self.critic_model_1(inps)

    @tf.function
    def update_targets(self):
        model_target_pairs = [
            (self.critic_model_0, self.target_critic_model_0),
            (self.critic_model_1, self.target_critic_model_1),
            (self.policy_model, self.target_policy_model),
        ]
        for model, target in model_target_pairs:
            for model_var, target_var in zip(model.trainable_variables, target.trainable_variables):
                target_var.assign(
                    (1 - self.tau) * target_var +
                    self.tau * model_var
                )

    @tf.function
    def train_critic(self, critic_states, actions, targets):
        with tf.GradientTape() as tape:
            estimates_0, estimates_1 = self.get_return_estimates(critic_states, actions, mode='both') # [batch_size, n_actions_in_movement, 1]
            loss_critic = 0.5 * (
                keras.losses.Huber()(estimates_0, tf.stop_gradient(targets)[..., tf.newaxis]) +
                keras.losses.Huber()(estimates_1, tf.stop_gradient(targets)[..., tf.newaxis])
            )
            vars = self.critic_model_0.variables + self.critic_model_1.variables
            grads = tape.gradient(loss_critic, vars)
            self.critic_optimizer.apply_gradients(zip(grads, vars))
        return loss_critic

    @tf.function
    def train_policy(self, policy_states, critic_states):
        with tf.GradientTape() as tape:
            # policy_states has shape [batch_size, state_size]
            # critic_states has shape [batch_size, n_actions_in_movement, state_size]
            actions, _, _ = self.get_actions(policy_states) # shape [batch_size, n_actions_in_movement, action_size]
            estimates = self.get_return_estimates(critic_states, actions, mode='mean')
            loss = - tf.reduce_sum(estimates)
            vars = self.policy_model.trainable_variables
            grads = tape.gradient(loss, vars)
            self.policy_optimizer.apply_gradients(zip(grads, vars))
        return loss

    @tf.function
    def train(self, policy_states, critic_states, actions, critic_target,
            policy=True, critic=True):
        losses = {}
        if critic:
            critic_loss = self.train_critic(
                critic_states, actions, critic_target)
            losses["critic"] = critic_loss
        if policy:
            policy_loss = self.train_policy(policy_states, critic_states)
            losses["policy"] = policy_loss
        self.update_targets()
        return losses

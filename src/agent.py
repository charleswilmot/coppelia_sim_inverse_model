import tensorflow as tf
from tensorflow import keras
from ornstein_uhlenbeck import OUProcess


def model_copy(model, fake_inp):
    clone = model.clone_model()
    fake_out = model(fake_inp)
    fake_out = clone(fake_inp)
    for model_var, clone_var in zip(model.variables, clone.variables):
        clone_var.assign(model_var)
    return clone


class Agent(object):
    def __init__(self,
            policy_learning_rate, policy_model_arch,
            critic_learning_rate, critic_model_arch,
            exploration, state_size, action_size, goal_size):
        #   POLICY
        self.policy_learning_rate = policy_learning_rate
        self.policy_model = keras.models.model_from_yaml(
            policy_model_arch.pretty())
        fake_inp = np.zeros(
            shape=(1, state_size + goal_size),
            dtype=np.float32
        )
        self.target_policy_model = model_copy(self.policy_model, fake_inp)
        self.policy_optimizer = keras.optimizers.Adam(self.policy_learning_rate)
        #   CRITIC
        self.critic_learning_rate = critic_learning_rate
        self.critic_model_0 = keras.models.model_from_yaml(
            critic_model_arch.pretty())
        self.critic_model_1 = self.critic_model_0.clone_model()
        fake_inp = np.zeros(
            shape=(1, state_size + goal_size + action_size),
            dtype=np.float32
        )
        self.target_critic_model_0 = model_copy(self.critic_model_0, fake_inp)
        self.target_critic_model_1 = model_copy(self.critic_model_1, fake_inp)
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)
        #   FORWARD
        self.forward_learning_rate = forward_learning_rate
        self.forward_model = keras.models.model_from_yaml(
            forward_model_arch.pretty())
        self.forward_optimizer = keras.optimizers.Adam(self.forward_learning_rate)
        #   EXPLORATION NOISE
        self.exploration_params = exploration
        self.exploration_type = exploration.type
        self.exploration_stddev = exploration.stddev
        self.exploration_damping = exploration.damping
        self.exploration_shape = exploration.shape
        self.ou_process = OUProcess(
            shape=self.exploration_shape,
            damping=self.exploration_damping,
            stddev=self.exploration_stddev
        )
        self._action_shape = action_shape.to_container(resolve=True)

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
    def get_noise(self):
        return self.ou_process()

    @tf.function
    def get_actions(self, states, goals, exploration=False):
        pure_actions = self.policy_model(states, goals)
        if exploration:
            noises = self.get_noise()
            noisy_actions = tf.clip_by_value(
                pure_actions + noises,
                clip_value_min=-1,
                clip_value_max=1
            )
            noises = noisy_actions - pure_actions
            return pure_actions, noisy_actions, noises
        else:
            return pure_actions

    @tf.function
    def get_predictions(self, states, actions):
        inps = tf.concat([states, actions], axis=-1)
        return self.forward_model(inps)

    @tf.function
    def get_return_estimate(self, states, actions, goals, target=False):
        inps = tf.concat([states, actions, goals])
        if target:
            target_0 = self.target_critic_model_0(inps)
            target_1 = self.target_critic_model_1(inps)
            return tf.minimum(target_0, target_1)
        else:
            return (self.critic_model_0(inps) + self.critic_model_1(inps)) / 2

    @tf.function
    def train_critic(self, states, actions, goals, targets):
        with tf.GradientTape() as tape:
            estimates = self.get_return_estimate(states, actions, goals)
            loss = keras.losses.Huber()(estimates, tf.stop_gradient(targets))
            vars = self.critic_model_0.variables + self.critic_model_1.variables
            grads = tape.gradient(loss, vars)
            self.critic_optimizer.apply_gradients(zip(grads, vars))
        return loss

    @tf.function
    def train_policy(self, states, goals):
        with tf.GradientTape() as tape:
            actions = self.get_actions(states, goals, exploration=False)
            estimates = self.get_return_estimate(states, actions, goals)
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
                    (1 - self.td3_tau) * target_var +
                    self.td3_tau * model_var
                )

    @tf.function
    def train(self, states, actions, goals, critic_target, forward_target,
            policy=True, critic=True, forward=True):
        if critic:
            critic_loss = self.train_critic(states, actions, goals, critic_target)
        if forward:
            forward_loss = self.train_forward(states, actions, forward_target)
        if policy:
            policy_loss = self.train_policy(states, goals)
        self.update_targets()
        return {
            "critic_loss": critic_loss,
            "forward_loss": forward_loss,
            "policy_loss": policy_loss,
        }

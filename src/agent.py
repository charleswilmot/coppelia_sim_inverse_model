import tensorflow as tf
from tensorflow import keras
from model import Model
from ornstein_uhlenbeck import OUProcess


class Agent(object):
    def __init__(self,
            policy_learning_rate, policy_model_arch,
            critic_learning_rate, critic_model_arch,
            exploration, prediction_time_window, action_shape):
        #   POLICY
        self.policy_learning_rate = policy_learning_rate
        self.policy_model_arch = policy_model_arch
        self.policy_model = Model(**policy_model_arch)
        self.policy_optimizer = keras.optimizers.Adam(self.policy_learning_rate)
        #   CRITIC
        self.critic_learning_rate = critic_learning_rate
        self.critic_model_arch = critic_model_arch
        self.critic_model = Model(**critic_model_arch)
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)
        #   EXPLORATION NOISE
        self.exploration_params = exploration
        self.exploration_type = exploration.type
        self.exploration_stddev = exploration.stddev
        self.exploration_damping = exploration.damping
        self.ou_process = None
        self._action_shape = action_shape.to_container(resolve=True)

    def save_weights(self, path):
        self.policy_model.save_weights(path + "/policy_model")
        self.critic_model.save_weights(path + "/critic_model")

    def load_weights(self, path):
        self.policy_model.load_weights(path + "/policy_model")
        self.critic_model.load_weights(path + "/critic_model")

    @tf.function
    def get_noise(self):
        if self.exploration_type == "ornstein_uhlenbeck":
            if self.ou_process is None:
                self.ou_process = OUProcess(
                    shape=self._action_shape,
                    damping=self.exploration_damping,
                    stddev=self.exploration_stddev
                )
            return self.ou_process()
        else:
            raise ValueError("Unrecognized exploration type ({})".format(
                self.exploration_type
            ))

    @tf.function
    def get_actions(self, states, goals, exploration=False):
        pure_actions = self.policy_model(states, goals)
        if exploration:
            noises = self.get_noise()
            noisy_actions = pure_actions + noises
            return pure_actions, noisy_actions, noises
        else:
            return pure_actions

    @tf.function
    def get_predictions(self, states, goals, logits=False):
        return self.critic_model(states, goals, logits=logits)

    @tf.function
    def train_critic(self, states, goals, targets):
        with tf.GradientTape() as tape:
            logits = self.get_predictions(
                states,
                goals,
                logits=True
            )
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets,
                logits=logits
            )
            loss = tf.reduce_sum( # batch dim
                tf.reduce_mean(   # component dim
                    losses,
                    axis=-1
                )
            )
            vars = self.critic_model.variables
            grads = tape.gradient(loss, vars)
            self.critic_optimizer.apply_gradients(zip(grads, vars))
        return loss

    @tf.function
    def train_policy(self, states, goals, noisy_actions):
        with tf.GradientTape() as tape:
            actions = self.get_actions(states, goals, exploration=False)
            losses = (actions - noisy_actions) ** 2
            loss = tf.reduce_sum( # batch dim
                tf.reduce_mean(   # component dim
                    losses,
                    axis=-1
                )
            )
            vars = self.policy_model.variables
            grads = tape.gradient(loss, vars)
            self.policy_optimizer.apply_gradients(zip(grads, vars))
        return loss

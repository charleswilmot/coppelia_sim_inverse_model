import tensorflow as tf
from tensorflow import keras
import numpy as np
from custom_layers import custom_objects, NormalNoise
from td3 import TD3


class Agent(object):
    def __init__(self,
            policy_primitive_learning_rate, policy_movement_learning_rate, policy_model_arch,
            critic_learning_rate, critic_model_arch,
            target_smoothing_stddev, tau, exploration_prob,
            state_size, action_size, goal_size, n_simulations,
            movement_exploration_prob_ratio,
            policy_bottleneck_size, policy_default_layer_size, critic_default_layer_size):
        self.movement_exploration_prob_ratio = movement_exploration_prob_ratio
        full_policy_model = keras.models.model_from_yaml(
            policy_model_arch.pretty(resolve=True),
            custom_objects=custom_objects
        )
        if not isinstance(full_policy_model.layers[-1], NormalNoise):
            raise ValueError("Last layer of the policy must be of type NormalNoise")
        noise_layers_indices = [
            i for i, layer in enumerate(full_policy_model.layers)
            if isinstance(layer, NormalNoise)
        ]
        if len(noise_layers_indices) > 2:
            raise ValueError("More than 2 NormalNoise layers have been found in the policy")
        self.has_movement_primitive = len(noise_layers_indices) == 2
        if self.has_movement_primitive:
            primitive_policy_model = keras.models.Sequential(
                full_policy_model.layers[
                :noise_layers_indices[0] + 1
            ])
            movement_policy_model = keras.models.Sequential(
                full_policy_model.layers[
                noise_layers_indices[0] + 1:noise_layers_indices[1] + 1
            ])
            self.primitive_size = primitive_policy_model.layers[-2].units
            self.primitive_td3 = TD3(
                policy_learning_rate=policy_primitive_learning_rate,
                policy_model=primitive_policy_model,
                critic_learning_rate=critic_learning_rate,
                critic_model=keras.models.model_from_yaml(
                    critic_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                ),
                target_smoothing_stddev=target_smoothing_stddev,
                tau=tau,
                policy_state_size=state_size + goal_size,
                critic_state_size=state_size + goal_size,
                action_size=self.primitive_size,
                n_simulations=n_simulations,
            )
            self.movement_td3 = TD3(
                policy_learning_rate=policy_movement_learning_rate,
                policy_model=movement_policy_model,
                critic_learning_rate=critic_learning_rate,
                critic_model=keras.models.model_from_yaml(
                    critic_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                ),
                target_smoothing_stddev=target_smoothing_stddev,
                tau=tau,
                policy_state_size=self.primitive_size,
                critic_state_size=state_size + goal_size,
                action_size=int(action_size),
                n_simulations=n_simulations,
            )
        else:
            movement_policy_model = full_policy_model
            self.primitive_td3 = None
            self.primitive_size = None
            self.movement_td3 = TD3(
                policy_learning_rate=policy_movement_learning_rate,
                policy_model=movement_policy_model,
                critic_learning_rate=critic_learning_rate,
                critic_model=keras.models.model_from_yaml(
                    critic_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                ),
                target_smoothing_stddev=target_smoothing_stddev,
                tau=tau,
                policy_state_size=state_size + goal_size,
                critic_state_size=state_size + goal_size,
                action_size=int(action_size),
                n_simulations=n_simulations,
            )
        self.n_simulations = n_simulations
        self.exploration_prob = exploration_prob

    def save_weights(self, path):
        self.movement_td3.save_weights(path + "/movement_td3")
        if self.has_movement_primitive:
            self.primitive_td3.save_weights(path + "/primitive_td3")

    def load_weights(self, path):
        self.movement_td3.load_weights(path + "/movement_td3")
        if self.has_movement_primitive:
            self.primitive_td3.load_weights(path + "/primitive_td3")

    @tf.function
    def get_primitive_and_movement(self, policy_states, target=False, noise=True):
        # policy_states shape [..., state_size]
        if self.has_movement_primitive:
            who_explores = tf.random.uniform(shape=(self.n_simulations,)) < self.movement_exploration_prob_ratio
            explore = tf.random.uniform(shape=(self.n_simulations,)) < self.exploration_prob
            mps_explore = tf.math.logical_and(tf.math.logical_not(who_explores), explore)
            movement_explore = tf.math.logical_and(who_explores, explore)
            pure_primitive, noisy_primitive, noise_primitive = self.primitive_td3.get_actions(policy_states, target=target, explore=mps_explore)
            pure_primitive = pure_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            noisy_primitive = noisy_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            noise_primitive = noise_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            primitive_passed_to_movement_net = noisy_primitive if noise else pure_primitive
            pure_movement, noisy_movement, noise_movement = self.movement_td3.get_actions(
                primitive_passed_to_movement_net, # shape [..., primitive_size]
                target=target,
                explore=movement_explore
            ) # shape [..., n_actions_in_movement, action_size]
            # pure_movement, noisy_movement, noise_movement = self.get_movement_from_primitive(
            #     primitive_passed_to_movement_net,
            #     target=target,
            #     explore=movement_explore
            # )
            return pure_primitive, noisy_primitive, noise_primitive, pure_movement, noisy_movement, noise_movement
        else:
            raise ValueError("Can not get MPs, Agent has no movement primitive")

    @tf.function
    def get_movement_from_primitive(self, primitives, target=False, explore=False):
        return self.movement_td3.get_actions(
            primitives, # shape [..., primitive_size]
            target=target,
            explore=explore
        ) # shape [..., n_actions_in_movement, action_size]

    @tf.function
    def get_primitive(self, policy_states, target=False):
        # policy_states shape [..., state_size]
        if self.has_movement_primitive:
            pure_primitive, noisy_primitive, noise_primitive = self.primitive_td3.get_actions(policy_states, target=target, explore=False)
            pure_primitive = pure_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            noisy_primitive = noisy_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            noise_primitive = noise_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            return pure_primitive, noisy_primitive, noise_primitive
        else:
            raise ValueError("Can not get MPs, Agent has no movement primitive")

    @tf.function
    def get_movement(self, policy_states, target=False):
        explore = tf.random.uniform(shape=(self.n_simulations,)) < self.exploration_prob
        return self.movement_td3.get_actions(policy_states, target=target, explore=explore)

    @tf.function
    def get_movement_return_estimates(self, critic_states, movement, target=False):
        return self.movement_td3.get_return_estimates(critic_states, movement, target=target)

    @tf.function
    def get_primitive_return_estimates(self, critic_states, mps, target=False):
        if self.has_movement_primitive:
            return self.primitive_td3.get_return_estimates(critic_states, mps, target=target)
        else:
            raise ValueError("Can not get MPs return estimate, Agent has no movement primitive")

    @tf.function
    def train_movement(self, policy_states, critic_states, movement, critic_target, policy=True, critic=True):
        return self.movement_td3.train(policy_states, critic_states, movement, critic_target, policy=policy, critic=critic)

    @tf.function
    def train_primitive(self, policy_states, critic_states, mps, critic_target, policy=True, critic=True):
        if self.has_movement_primitive:
            return self.primitive_td3.train(policy_states, critic_states, mps, critic_target, policy=policy, critic=critic)
        else:
            raise ValueError("Can not train MPs, Agent has no movement primitive")

    def set_movement_log_stddevs(self, values):
        self.movement_td3.set_log_stddevs(values)

    def get_movement_log_stddevs(self):
        return self.movement_td3.get_log_stddevs()

    def set_primitive_log_stddevs(self, values):
        self.primitive_td3.set_log_stddevs(values)

    def get_primitive_log_stddevs(self):
        return self.primitive_td3.get_log_stddevs()

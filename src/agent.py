import tensorflow as tf
from tensorflow import keras
import numpy as np
from custom_layers import custom_objects, PrimitiveModelEnd, BottleneckExploration
from td3 import TD3


class Agent(object):
    def __init__(self,
            policy_primitive_learning_rate, policy_movement_learning_rate,
            primitive_exploration_stddev, movement_exploration_stddev,
            policy_model_arch, critic_learning_rate, critic_model_arch,
            target_smoothing_stddev, tau, exploration_prob,
            state_size, action_size, goal_size, n_simulations,
            movement_exploration_prob_ratio,
            policy_bottleneck_size, policy_default_layer_size, critic_default_layer_size):
        self.movement_exploration_prob_ratio = movement_exploration_prob_ratio
        self.primitive_exploration_stddev = primitive_exploration_stddev
        self.movement_exploration_stddev = movement_exploration_stddev
        full_policy_model = keras.models.model_from_yaml(
            policy_model_arch.pretty(resolve=True),
            custom_objects=custom_objects
        )
        bottleneck_exploration_indices = [
            i for i, layer in enumerate(full_policy_model.layers)
            if isinstance(layer, BottleneckExploration)
        ]
        if len(bottleneck_exploration_indices) > 1:
            raise ValueError("More than 1 BottleneckExploration layer has been found in the policy")
        if len(bottleneck_exploration_indices) == 1:
            self.has_bottleneck_exploration = True
            bottleneck_exploration_index = bottleneck_exploration_indices[0]
            self.movement_policy_model_0 = keras.models.Sequential(
                full_policy_model.layers[:bottleneck_exploration_index])
            self.movement_policy_model_1 = keras.models.Sequential(
                full_policy_model.layers[bottleneck_exploration_index + 1:])
            print('#### found bottleneck exploration layer at index ', bottleneck_exploration_index)
        else:
            self.has_bottleneck_exploration = False
        primitive_model_end_indices = [
            i for i, layer in enumerate(full_policy_model.layers)
            if isinstance(layer, PrimitiveModelEnd)
        ]
        if len(primitive_model_end_indices) > 1:
            raise ValueError("More than 1 PrimitiveModelEnd layer has been found in the policy")
        self.has_movement_primitive = len(primitive_model_end_indices) == 1
        if self.has_movement_primitive:
            primitive_model_end_index = primitive_model_end_indices[0]
            primitive_policy_model = keras.models.Sequential(
                full_policy_model.layers[:primitive_model_end_index])
            movement_policy_model = keras.models.Sequential(
                full_policy_model.layers[primitive_model_end_index + 1:])
            self.primitive_size = primitive_policy_model.layers[-1].units
            self.primitive_td3 = TD3(
                policy_learning_rate=policy_primitive_learning_rate,
                policy_model=primitive_policy_model,
                critic_learning_rate=critic_learning_rate,
                critic_model=keras.models.model_from_yaml(
                    critic_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                ),
                exploration_stddev=primitive_exploration_stddev,
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
                exploration_stddev=movement_exploration_stddev,
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
                exploration_stddev=movement_exploration_stddev,
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

    def load_weights(self, path,
            mvt_policy_model=True, mvt_target_policy_model=True, mvt_critic_model_0=True,
            mvt_critic_model_1=True, mvt_target_critic_model_0=True, mvt_target_critic_model_1=True,
            prim_policy_model=True, prim_target_policy_model=True, prim_critic_model_0=True,
            prim_critic_model_1=True, prim_target_critic_model_0=True, prim_target_critic_model_1=True):
        d = {
            "mvt_policy_model": mvt_policy_model,
            "mvt_target_policy_model": mvt_target_policy_model,
            "mvt_critic_model_0": mvt_critic_model_0,
            "mvt_critic_model_1": mvt_critic_model_1,
            "mvt_target_critic_model_0": mvt_target_critic_model_0,
            "mvt_target_critic_model_1": mvt_target_critic_model_1,
            "prim_policy_model": prim_policy_model,
            "prim_target_policy_model": prim_target_policy_model,
            "prim_critic_model_0": prim_critic_model_0,
            "prim_critic_model_1": prim_critic_model_1,
            "prim_target_critic_model_0": prim_target_critic_model_0,
            "prim_target_critic_model_1": prim_target_critic_model_1,
        }
        which = " ".join([key for key, value in d.items() if value])
        print("[agent] loading weights ({})".format(which))
        self.movement_td3.load_weights(path + "/movement_td3",
            mvt_policy_model, mvt_target_policy_model, mvt_critic_model_0,
            mvt_critic_model_1, mvt_target_critic_model_0, mvt_target_critic_model_1)
        if self.has_movement_primitive:
            self.primitive_td3.load_weights(path + "/primitive_td3",
                prim_policy_model, prim_target_policy_model, prim_critic_model_0,
                prim_critic_model_1, prim_target_critic_model_0, prim_target_critic_model_1)

    @tf.function
    def get_primitive_and_movement(self, policy_states, target=False, noise=True):
        # policy_states shape [..., state_size]
        if self.has_movement_primitive:
            who_explores = tf.random.uniform(shape=(self.n_simulations,)) < self.movement_exploration_prob_ratio
            explore = tf.random.uniform(shape=(self.n_simulations,)) < self.exploration_prob
            primitive_explore = tf.math.logical_and(tf.math.logical_not(who_explores), explore)
            movement_explore = tf.math.logical_and(who_explores, explore)
            pure_primitive, noisy_primitive, noise_primitive = self.primitive_td3.get_actions(policy_states, target=target, explore=primitive_explore)
            pure_primitive = pure_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            noisy_primitive = noisy_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            noise_primitive = noise_primitive[..., 0, :] # sequence of ONE primitive --> only one primitive. Resulting shape [batch_size, primitive_size]
            # primitive_passed_to_movement_net = noisy_primitive if noise else pure_primitive
            # pure_movement, noisy_movement, noise_movement = self.movement_td3.get_actions(
            #     primitive_passed_to_movement_net, # shape [..., primitive_size]
            #     target=target,
            #     explore=movement_explore
            # ) # shape [..., n_actions_in_movement, action_size]
            if noise:
                pure_movement, noisy_movement, noise_movement = self.movement_td3.get_actions(
                    noisy_primitive, # shape [..., primitive_size]
                    target=target,
                    explore=movement_explore
                ) # shape [..., n_actions_in_movement, action_size]
                pure_movement_pure_primitive, noisy_movement_pure_primitive, noise_movement_pure_primitive = self.movement_td3.get_actions(
                    pure_primitive, # shape [..., primitive_size]
                    target=target,
                    explore=movement_explore
                ) # shape [..., n_actions_in_movement, action_size]
                noise_movement = noisy_movement - pure_movement_pure_primitive
            else:
                pure_movement, noisy_movement, noise_movement = self.movement_td3.get_actions(
                    pure_primitive, # shape [..., primitive_size]
                    target=target,
                    explore=movement_explore
                ) # shape [..., n_actions_in_movement, action_size]
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
        if self.has_bottleneck_exploration:
            new_shape = tf.concat([tf.shape(policy_states)[:-1], [-1], [self.movement_td3.action_size]], axis=0)
            who_explores = tf.random.uniform(shape=(self.n_simulations,)) < self.movement_exploration_prob_ratio
            primitive_explore = tf.math.logical_and(tf.math.logical_not(who_explores), explore)
            movement_explore = tf.math.logical_and(who_explores, explore)
            bn = self.movement_policy_model_0(policy_states)
            noisy_bn = tf.clip_by_value(bn + tf.random.truncated_normal(
                shape=tf.shape(bn),
                stddev=self.primitive_exploration_stddev,
            ) * tf.cast(tf.reshape(primitive_explore, [-1, 1]), tf.float32), -1, 1)
            noisy_out = tf.reshape(self.movement_policy_model_1(noisy_bn), new_shape)
            noisy_out += tf.random.truncated_normal(
                shape=tf.shape(noisy_out),
                stddev=self.movement_exploration_stddev,
            ) * tf.cast(tf.reshape(movement_explore, [-1, 1]), tf.float32)
            out = tf.reshape(self.movement_policy_model_1(bn), new_shape)
            noise = noisy_out - out
            return out, noisy_out, noise
        else:
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

import numpy as np
from buffer import Buffer
from agent import Agent
from simulation import SimulationPool, MODEL_PATH
from tensorflow.keras.metrics import Mean
import tensorflow as tf
import time
import os
from visualization import Visualization
from collections import OrderedDict
from tensorboard.plugins.hparams import api as hp
from imageio import get_writer


def _rec_expected_time(probs, t=1):
    if not probs.shape[-1]:
        return t
    else:
        return t * probs[..., 0] + \
               (1 - probs[..., 0]) * _rec_expected_time(probs[..., 1:], t=t+1)


def get_time_prediction(prediction, goal):
    probs = np.prod(2 * goal * prediction + 1 - goal - prediction, axis=-1)
    return _rec_expected_time(probs)


class Procedure(object):
    def __init__(self, agent_conf, buffer_conf, simulation_conf, procedure_conf):
        #   PROCEDURE CONF
        self.episode_length = procedure_conf.episode_length
        self.critic_updates_per_sample = procedure_conf.critic_updates_per_sample
        self.policy_updates_per_sample = procedure_conf.policy_updates_per_sample
        self.forward_updates_per_sample = procedure_conf.forward_updates_per_sample
        self.batch_size = procedure_conf.batch_size
        self.n_simulations = simulation_conf.n
        self.movement_mode = procedure_conf.movement_mode
        self.movement_span = int(procedure_conf.movement_span_in_sec / \
                                 procedure_conf.simulation_timestep)
        self.movement_modes = [
            self.movement_mode for i in range(self.n_simulations)
        ]
        self.movement_spans = [
            self.movement_span for i in range(self.n_simulations)
        ]
        #    HPARAMS
        self._hparams = OrderedDict([
            ("policy_LR", agent_conf.policy_learning_rate),
            ("critic_LR", agent_conf.critic_learning_rate),
            ("buffer", buffer_conf.size),
            ("policy_update_rate", procedure_conf.policy_updates_per_sample),
            ("critic_update_rate", procedure_conf.critic_updates_per_sample),
            ("forward_update_rate", procedure_conf.forward_updates_per_sample),
            ("ep_length", procedure_conf.episode_length),
            ("batch_size", procedure_conf.batch_size),
            ("noise_type", agent_conf.exploration.type),
            ("noise_std", agent_conf.exploration.stddev),
            ("noise_damp", agent_conf.exploration.damping),
            ("movement_mode", procedure_conf.movement_mode),
            ("movement_span", procedure_conf.movement_span_in_sec),
        ])
        #   VISUALIZATION
        if procedure_conf.visualize:
            self._visualization = Visualization(
                mini=0,
                maxi=agent_conf.prediction_time_window + 10
            )
        else:
            self._visualization = None
        #   OBJECTS
        self.agent = Agent(**agent_conf)
        self.buffer = Buffer(**buffer_conf)
        #   SIMULATION POOL
        guis = simulation_conf.guis.to_container(resolve=True)
        self.simulation_pool = SimulationPool(
            simulation_conf.n,
            scene=MODEL_PATH + '/custom_timestep.ttt',
            guis=guis
        )
        self.simulation_pool.set_simulation_timestep(
            procedure_conf.simulation_timestep
        )
        self.simulation_pool.create_environment(simulation_conf.environment)
        self.simulation_pool.set_reset_poses()
        self.simulation_pool.set_control_loop_enabled(False)
        self.simulation_pool.start_sim()
        self.simulation_pool.step_sim()
        print("[procedure] all simulation started")
        with self.simulation_pool.specific(0):
            self.goal_size = self.simulation_pool.get_n_registers()[0]
            self.state_size = self.simulation_pool.get_state()[0].shape[0]
            self.action_size = self.simulation_pool.get_n_joints()[0]
            self.prediction_size = (
                agent_conf.prediction_time_window,
                simulation_conf.n_register
            )

        print("self.goal_size", self.goal_size)
        print("self.state_size", self.state_size)
        print("self.action_size", self.action_size)
        print("self.prediction_size", self.prediction_size)

        #   DEFINING DATA BUFFERS
        # policy
        self._data_type = np.dtype([
            ("states", np.float32, self.state_size),
            ("next_states", np.float32, self.state_size),
            ("goals", np.float32, self.goal_size),
            ("current_goals", np.float32, self.goal_size),
            ("pure_actions", np.float32, self.action_size),
            ("noisy_actions", np.float32, self.action_size),
            ("predictions", np.float32, self.prediction_size),
            ("return", np.float32)
            ("_simulator_index", np.int16)
        ])
        self._data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=self._data_type
        )
        # COUNTERS
        self.n_exploration_episodes = 0
        self.n_evaluation_episodes = 0
        self.n_transition_gathered = 0
        self.n_policy_training = 0
        self.n_critic_training = 0
        self.n_forward_training = 0
        self.n_global_training = 0

        # TENSORBOARD LOGGING
        self.tb_training_policy_loss = Mean(
            "training/policy_loss",
            dtype=tf.float32
        )
        self.tb_training_critic_loss = Mean(
            "training/critic_loss",
            dtype=tf.float32
        )
        self.tb_training_forward_loss = Mean(
            "training/forward_loss",
            dtype=tf.float32
        )
        self.tb_collection_exploration_it_per_sec = Mean(
            "collection/exploration_it_per_sec",
            dtype=tf.float32
        )
        self.tb_collection_evaluation_it_per_sec = Mean(
            "collection/evaluation_it_per_sec",
            dtype=tf.float32
        )
        self.tb_collection_exploration_success_rate_percent = Mean(
            "collection/exploration_success_rate_percent",
            dtype=tf.float32
        )
        self.tb_collection_evaluation_success_rate_percent = Mean(
            "collection/evaluation_success_rate_percent",
            dtype=tf.float32
        )
        self.tb_collection_exploration_diversity_per_ep = Mean(
            "collection/exploration_diversity_per_ep",
            dtype=tf.float32
        )
        self.tb_collection_evaluation_diversity_per_ep = Mean(
            "collection/evaluation_diversity_per_ep",
            dtype=tf.float32
        )
        self.tb_collection_exploration_delta_distance_to_goal = Mean(
            "collection/exploration_delta_distance_to_goal",
            dtype=tf.float32
        )
        self.tb_collection_evaluation_delta_distance_to_goal = Mean(
            "collection/evaluation_delta_distance_to_goal",
            dtype=tf.float32
        )
        self.tb_collection_exploration_n_register_change = Mean(
            "collection/exploration_n_register_change",
            dtype=tf.float32
        )
        self.tb_collection_evaluation_n_register_change = Mean(
            "collection/evaluation_n_register_change",
            dtype=tf.float32
        )
        self.tb_collection_exploration_one_away_sucess_rate = Mean(
            "collection/exploration_one_away_sucess_rate",
            dtype=tf.float32
        )
        self.tb_collection_evaluation_one_away_sucess_rate = Mean(
            "collection/evaluation_one_away_sucess_rate",
            dtype=tf.float32
        )
        self.summary_writer = tf.summary.create_file_writer("logs")
        with self.summary_writer.as_default():
            hp.hparams(self._hparams)
        # TREE STRUCTURE
        os.makedirs('./replays', exist_ok=True)

    def dump_buffers(self):
        os.makedirs('./buffers', exist_ok=True)
        path = "./buffers/critic_{:6d}.pkl".format(self.n_critic_training)
        self.critic_buffer.dump(path)
        path = "./buffers/policy_{:6d}.pkl".format(self.n_policy_training)
        self.policy_buffer.dump(path)

    def log_metric_list(self, metric_list, step):
        with self.summary_writer.as_default():
            for metric in training_critic_metric_list:
                tf.summary.scalar(metric.name, metric.result(), step=step)
                metric.reset_states()

    def log_summaries(self, exploration=True, evaluation=True, critic=True,
            policy=True, forward=True):
        training_critic_metric_list = [
            self.tb_training_critic_loss
        ]
        training_policy_metric_list = [
            self.tb_training_policy_loss
        ]
        training_forward_metric_list = [
            self.tb_training_forward_loss
        ]
        collection_exploration_metric_list = [
            self.tb_collection_exploration_it_per_sec,
            self.tb_collection_exploration_success_rate_percent,
            self.tb_collection_exploration_diversity_per_ep,
            self.tb_collection_exploration_delta_distance_to_goal,
            self.tb_collection_exploration_n_register_change,
            self.tb_collection_exploration_one_away_sucess_rate,
        ]
        collection_evaluation_metric_list = [
            self.tb_collection_evaluation_it_per_sec,
            self.tb_collection_evaluation_success_rate_percent,
            self.tb_collection_evaluation_diversity_per_ep,
            self.tb_collection_evaluation_delta_distance_to_goal,
            self.tb_collection_evaluation_n_register_change,
            self.tb_collection_evaluation_one_away_sucess_rate,
        ]
        if exploration:
            self.log_metric_list(
                collection_exploration_metric_list,
                self.n_exploration_episodes
            )
        if evaluation:
            self.log_metric_list(
                collection_evaluation_metric_list,
                self.n_evaluation_episodes
            )
        if critic:
            self.log_metric_list(
                training_critic_metric_list,
                self.n_critic_training
            )
        if policy:
            self.log_metric_list(
                training_policy_metric_list,
                self.n_policy_training
            )
        if forward:
            self.log_metric_list(
                training_forward_metric_list,
                self.n_forward_training
            )

    def _get_current_training_ratio(self):
        if self.n_transition_gathered != 0:
            return self.n_global_training * \
                    self.batch_size / \
                    self.n_transition_gathered
        else:
            return np.inf
    current_training_ratio = property(_get_current_training_ratio)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        self.simulation_pool.close()

    def save(self):
        """Saves the model in the appropriate directory"""
        path = "./checkpoints/{:08d}".format(self.n_policy_episodes)
        self.agent.save_weights(path)

    def restore(self, path):
        """Restores the weights from a checkpoint"""
        self.agent.load_weights(path)

    def sample_goals(self, n=None):
        n = self.n_simulations if n is None else n
        """Returns a binary vector corresponding to the goal states of the
        actuators in the simulation for each simulation"""
        return np.random.randint(2, size=(n, self.goal_size))

    def reset_simulations(self):
        goals = self.sample_goals()
        with self.simulation_pool.distribute_args():
            states, current_goals = \
                tuple(zip(*self.simulation_pool.reset(goals)))
        return np.vstack(states), np.vstack(current_goals)

    def replay(self, exploration=False, record=False, n_episodes=10,
            video_name='replay.mp4', resolution=[320, 240]):
        """Applies the current policy in the environment"""
        with self.simulation_pool.specific(0):
            if record:
                writer = get_writer(video_name)
                cam_id = self.simulation_pool.add_camera(
                    position=(1, 1, 1),
                    orientation=(
                        24 * np.pi / 36,
                        -7 * np.pi / 36,
                         4 * np.pi / 36
                    ),
                    resolution=resolution
                )[0]
            for i in range(n_episodes):
                goals = self.sample_goals(1)
                states, current_goals = self.reset_simulations()
                for iteration in range(self.episode_length):
                    frame = self.simulation_pool.get_frame(cam_id)[0]
                    frame = (frame * 255).astype(np.uint8)
                    if record:
                        writer.append_data(frame)
                        if iteration == 0:
                            for i in range(24):
                                writer.append_data(frame)
                    if exploration:
                        _, noisy_actions, _ = self.agent.get_actions(
                            states, goals, exploration=True)
                        states, current_goals = self.apply_action(noisy_actions)
                    else:
                        pure_actions = self.agent.get_actions(
                            states, goals, exploration=False)
                        states, current_goals = self.apply_action(pure_actions)
                    # print("success?", (current_goals == goals).all())
            if record:
                writer.close()
                self.simulation_pool.delete_camera(cam_id)

    def collect_data(self, exploration=True):
        """Performs one episode of exploration, places data in the policy
        buffer"""
        goals = self.sample_goals()
        states, current_goals = self.reset_simulations()
        time_start = time.time()
        for iteration in range(self.episode_length):
            pure_actions, noisy_actions, noises = self.agent.get_actions(
                states, goals, exploration=True)
            predictions = self.agent.get_predictions(states, noisy_actions)
            next_actions = self.agent.get_actions(
                predictions, goals, exploration=False
            )
            next_return_estimate = self.agent.get_next_return_estimate(
                predictions,
                next_actions,
                goals
            ).numpy()
            indices_best = np.argmax(next_return_estimate, axis=-1)
            noisy_actions = noisy_actions[:, indices_best]
            next_states = predictions[:, indices_best]
            next_actions = next_actions[:, indices_best]
            self._data_buffer[:, iteration]["states"] = states
            self._data_buffer[:, iteration]["next_states"] = states
            self._data_buffer[:, iteration]["goals"] = goals
            self._data_buffer[:, iteration]["current_goals"] = current_goals
            self._data_buffer[:, iteration]["pure_actions"] = pure_actions
            self._data_buffer[:, iteration]["next_pure_actions"] = next_actions
            self._data_buffer[:, iteration]["noisy_actions"] = best_noisy_actions
            self._data_buffer[:, iteration]["predictions"] = predictions
            states, current_goals = self.apply_action(best_noisy_actions)
        # COMPUTE RETURN (TODO)
        pass
        # HINDSIGHT EXPERIENCE
        register_change = (
            self._policy_data_buffer["current_goals"][:-1] !=
            self._policy_data_buffer["current_goals"][1:]
        ).any(axis=-1)
        iteration_indices = {
            i: changes.nonzero()[0]
            for i, changes in enumerate(register_change.T) if changes.any()
        }
        for_hindsight = []
        prediction_error_sum = 0
        prediction_error_n = 0
        for simulation, changes in iteration_indices.items():
            for iteration in range(self._policy_data_buffer.shape[0]):
                next = np.argmax(changes > iteration)
                if next or changes[0] > iteration:
                    for transition_index in changes[next:]:
                        state = self._policy_data_buffer["states"][iteration, simulation]
                        actual_goal = self._policy_data_buffer["current_goals"][transition_index + 1, simulation]
                        prediction = self.agent.get_predictions(
                            state[np.newaxis],
                            actual_goal[np.newaxis]).numpy()[0]
                        prediction = get_time_prediction(
                            prediction,              # 10, 4
                            actual_goal[np.newaxis]  #  1, 4
                        )
                        truth = transition_index - iteration
                        prediction_error_sum += np.abs(prediction - np.clip(truth, 0, self.prediction_time_window))
                        prediction_error_n += 1
                        prediction = int(prediction)
                        if truth < prediction * (1 - self.pessimism):
                            print("YES: sim:{:2d} it:{:2d}  took me {:2d} steps to go from {} to {} at {:2d}, I predicted {:2d} ({:.3f})".format(
                                simulation,
                                iteration,
                                truth,
                                self._policy_data_buffer["current_goals"][transition_index, simulation],
                                actual_goal,
                                transition_index,
                                prediction,
                                prediction * (1 - self.pessimism)))
                            copy = np.copy(self._policy_data_buffer[iteration, simulation])
                            copy["goals"] = actual_goal
                            for_hindsight.append(copy)
                        else:
                            print("NO : sim:{:2d} it:{:2d}  took me {:2d} steps to go from {} to {} at {:2d}, I predicted {:2d} ({:.3f})".format(
                                simulation,
                                iteration,
                                truth,
                                self._policy_data_buffer["current_goals"][transition_index, simulation],
                                actual_goal,
                                transition_index,
                                prediction,
                                prediction * (1 - self.pessimism)))
        if len(for_hindsight):
            hindsight_data = np.vstack(for_hindsight)
            self.policy_buffer.integrate(hindsight_data)
            n_discoveries = len(hindsight_data)
        else:
            n_discoveries = 0
        self.n_policy_transition_gathered += n_discoveries
        self.n_policy_episodes += self.n_simulations
        time_stop = time.time()
        self.tb_collection_policy_it_per_sec(
            self.episode_length * self.n_simulations / (time_stop - time_start)
        )
        goal_reached = (
            self._policy_data_buffer["goals"] == \
            self._policy_data_buffer["current_goals"]
        ).all(axis=-1)
        self.tb_collection_policy_success_rate_percent(
            100 * np.mean(goal_reached.any(axis=0))
        )
        self.tb_collection_policy_discoveries_per_ep(
            n_discoveries
        )
        if prediction_error_n:
            self.tb_collection_policy_mean_abs_prediction_error_it(
                prediction_error_sum / prediction_error_n
            )
        n_uniques = 0
        for simulation in range(self.n_simulations):
            episode = self._policy_data_buffer["current_goals"][:, simulation]
            n_uniques += len(np.unique(episode, axis=0))
        self.tb_collection_policy_goal_diversity(
            n_uniques / self.n_simulations
        )
        distance_at_start = np.sqrt(np.sum((
            self._policy_data_buffer["current_goals"][0] -
            self._policy_data_buffer["goals"][0]
        ) ** 2, axis=-1))
        distance_at_end = np.sqrt(np.sum((
            self._policy_data_buffer["current_goals"][-1] -
            self._policy_data_buffer["goals"][-1]
        ) ** 2, axis=-1))
        self.tb_collection_policy_delta_distance_to_goal(
            np.mean(distance_at_start - distance_at_end)
        )
        self.tb_collection_policy_n_register_change(
            np.mean(np.sum(register_change, axis=0))
        )

    def get_data(self):
        states, current_goals = tuple(zip(*self.simulation_pool.get_data()))
        return np.vstack(states), np.vstack(current_goals)

    def apply_action(self, actions):
        with self.simulation_pool.distribute_args():
            states, current_goals = \
                tuple(zip(*self.simulation_pool.apply_movement(
                    actions,
                    mode=self.movement_modes,
                    span=self.movement_spans
                )))
        return np.vstack(states), np.vstack(current_goals)

    # def gather_critic_data(self):
    #     """Performs one episode of testing, places the data in the critic
    #     buffer"""
    #     time_start = time.time()
    #     goals = self.sample_goals()
    #     states, current_goals = self.reset_simulations()
    #     for iteration in range(self.episode_length):
    #         pure_actions = self.agent.get_actions(
    #             states, goals, exploration=False)
    #         predictions = self.agent.get_predictions(states, goals)
    #         self._critic_data_buffer[iteration]["states"] = states
    #         self._critic_data_buffer[iteration]["goals"] = goals
    #         self._critic_data_buffer[iteration]["current_goals"] = current_goals
    #         self._critic_data_buffer[iteration]["pure_actions"] = pure_actions
    #         self._critic_data_buffer[iteration]["predictions"] = predictions
    #         states, current_goals = self.apply_action(pure_actions)
    #     # COMPLETE THE BUFFER WITH THE PREDICTION TARGETS
    #     ptw = self.prediction_time_window
    #     for i in range(self.prediction_time_window):
    #         self._critic_data_buffer["targets"][:-ptw - 1, :, i] = \
    #             self._critic_data_buffer["current_goals"][i + 1:i + self.episode_length - ptw]
    #     valid_part_of_buffer = self._critic_data_buffer[:-ptw - 1]
    #     goal_reached = (
    #         self._critic_data_buffer["goals"] == \
    #         self._critic_data_buffer["current_goals"]
    #     ).all(axis=-1)
    #     time_stop = time.time()
    #     # ADD TO THE REPLAY BUFFER AND LOG
    #     self.critic_buffer.integrate(valid_part_of_buffer)
    #     self.n_critic_transition_gathered += (self.episode_length - ptw - 1) * self.n_simulations
    #     self.n_critic_episodes += self.n_simulations
    #     # LOGGING
    #     register_change = (
    #         self._critic_data_buffer["current_goals"][:-1] !=
    #         self._critic_data_buffer["current_goals"][1:]
    #     ).any(axis=-1)
    #     iteration_indices = {
    #         i: changes.nonzero()[0]
    #         for i, changes in enumerate(register_change.T) if changes.any()
    #     }
    #     prediction_error_sum = 0
    #     prediction_error_n = 0
    #     for simulation, changes in iteration_indices.items():
    #         for iteration in range(self._critic_data_buffer.shape[0]):
    #             next = np.argmax(changes > iteration)
    #             if next or changes[0] > iteration:
    #                 for transition_index in changes[next:]:
    #                     state = self._critic_data_buffer["states"][iteration, simulation]
    #                     actual_goal = self._critic_data_buffer["current_goals"][transition_index + 1, simulation]
    #                     prediction = self.agent.get_predictions(
    #                         state[np.newaxis],
    #                         actual_goal[np.newaxis]).numpy()[0]
    #                     prediction = get_time_prediction(
    #                         prediction,              # 10, 4
    #                         actual_goal[np.newaxis]  #  1, 4
    #                     )
    #                     truth = transition_index - iteration
    #                     prediction_error_sum += np.abs(prediction - np.clip(truth, 0, self.prediction_time_window))
    #                     prediction_error_n += 1
    #     if prediction_error_n:
    #         self.tb_collection_critic_mean_abs_prediction_error_it(
    #             prediction_error_sum / prediction_error_n
    #         )
    #     #
    #     one_away = np.sum(np.abs(
    #         self._critic_data_buffer["goals"] -
    #         self._critic_data_buffer["current_goals"]
    #     ), axis=-1) == 1
    #     one_away_successes = np.logical_and(one_away[:-1], goal_reached[1:])
    #     one_away_fails = np.logical_and(
    #         np.logical_and(one_away[:-1], np.logical_not(one_away[1:])),
    #         np.logical_not(goal_reached[1:])
    #     )
    #     n_one_away_success = np.sum(one_away_successes)
    #     n_one_away_fail = np.sum(one_away_fails)
    #     if n_one_away_success + n_one_away_fail:
    #         self.tb_collection_critic_one_away_sucess_rate(
    #             100 * n_one_away_success / (n_one_away_success + n_one_away_fail)
    #         )
    #     #
    #     self.tb_collection_critic_it_per_sec(
    #         self.episode_length * self.n_simulations / (time_stop - time_start)
    #     )
    #     #
    #     self.tb_collection_critic_mean_abs_prediction_error(
    #         np.mean(np.abs(
    #             self._critic_data_buffer["targets"] -
    #             self._critic_data_buffer["predictions"]
    #         ))
    #     )
    #     #
    #     successful_episode = goal_reached.any(axis=0)
    #     self.tb_collection_critic_success_rate_percent(
    #         100 * np.mean(successful_episode)
    #     )
    #     #
    #     n_uniques = 0
    #     for simulation in range(self.n_simulations):
    #         episode = self._critic_data_buffer["current_goals"][:, simulation]
    #         n_uniques += len(np.unique(episode, axis=0))
    #     self.tb_collection_critic_goal_diversity(
    #         n_uniques / self.n_simulations
    #     )
    #     distance_at_start = np.sqrt(np.sum((
    #         self._critic_data_buffer["current_goals"][0] -
    #         self._critic_data_buffer["goals"][0]
    #     ) ** 2, axis=-1))
    #     distance_at_end = np.sqrt(np.sum((
    #         self._critic_data_buffer["current_goals"][-1] -
    #         self._critic_data_buffer["goals"][-1]
    #     ) ** 2, axis=-1))
    #     #
    #     self.tb_collection_critic_delta_distance_to_goal(
    #         np.mean(distance_at_start - distance_at_end)
    #     )
    #     #
    #     self.tb_collection_critic_n_register_change(
    #         np.mean(np.sum(register_change, axis=0))
    #     )
    #     index_visualize = np.argmax(successful_episode) # index sucess episode if any
    #     # TODO:
    #     if self._visualization is not None:
    #         self._visualization.update_critic(
    #             target=self._critic_data_buffer["targets"],
    #             prediction=self._critic_data_buffer["predictions"],
    #         )

    def train(self, policy=True, critic=True, forward=True):
        data = self.buffer.sample(self.batch_size)
        losses = self.agent.train(
            data["states"],
            data["goals"],
            data["noisy_actions"]
        )
        if policy:
            self.n_policy_training += 1
            self.tb_training_policy_loss(losses["policy"])
        if critic:
            self.n_critic_training += 1
            self.tb_training_critic_loss(losses["critic"])
        if forward:
            self.n_forward_training += 1
            self.tb_training_forward_loss(losses["forward"])
        self.n_global_training += 1
        return losses

    def collect_and_train(self, policy=True, critic=True, forward=True):
        self.collect_data(exploration=True)
        while self.current_training_ratio < self.updates_per_sample:
            self.train(policy=policy, critic=critic, forward=forward)

    def collect_train_and_log(self, policy=True, critic=True, forward=True,
            evaluation=False):
        self.collect_and_train(policy=policy, critic=critic, forward=forward)
        if evaluation:
            self.collect_data(exploration=False)
        if self.n_critic_episodes % 1 == 0:
            self.log_summaries(exploration=True, evaluation=evaluation,
                policy=policy, critic=critic, forward=forward)

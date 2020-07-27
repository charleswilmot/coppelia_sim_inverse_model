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
        self.tb = {}
        self.tb["training"] = {}
        self.tb["training"]["policy"] = {}
        self.tb["training"]["policy"]["loss"] = Mean(
            "training/policy_loss", dtype=tf.float32)
        self.tb["training"]["critic"] = {}
        self.tb["training"]["critic"]["loss"] = Mean(
            "training/critic_loss", dtype=tf.float32)
        self.tb["training"]["forward"] = {}
        self.tb["training"]["forward"]["loss"] = Mean(
            "training/forward_loss", dtype=tf.float32)
        self.tb["collection"] = {}
        self.tb["collection"]["exploration"] = {}
        self.tb["collection"]["evaluation"] = {}
        self.tb["collection"]["exploration"]["it_per_sec"] = Mean(
            "collection/exploration_it_per_sec", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["it_per_sec"] = Mean(
            "collection/evaluation_it_per_sec", dtype=tf.float32)
        self.tb["collection"]["exploration"]["success_rate_percent"] = Mean(
            "collection/exploration_success_rate_percent", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["success_rate_percent"] = Mean(
            "collection/evaluation_success_rate_percent", dtype=tf.float32)
        self.tb["collection"]["exploration"]["diversity_per_ep"] = Mean(
            "collection/exploration_diversity_per_ep", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["diversity_per_ep"] = Mean(
            "collection/evaluation_diversity_per_ep", dtype=tf.float32)
        self.tb["collection"]["exploration"]["delta_distance_to_goal"] = Mean(
            "collection/exploration_delta_distance_to_goal", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["delta_distance_to_goal"] = Mean(
            "collection/evaluation_delta_distance_to_goal", dtype=tf.float32)
        self.tb["collection"]["exploration"]["n_register_change"] = Mean(
            "collection/exploration_n_register_change", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["n_register_change"] = Mean(
            "collection/evaluation_n_register_change", dtype=tf.float32)
        self.tb["collection"]["exploration"]["one_away_sucess_rate"] = Mean(
            "collection/exploration_one_away_sucess_rate", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["one_away_sucess_rate"] = Mean(
            "collection/evaluation_one_away_sucess_rate", dtype=tf.float32)
        #
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

    def log_metrics(self, key1, key2, step):
        with self.summary_writer.as_default():
            for name, metric in self.tb[key1][key2].items():
                tf.summary.scalar(metric.name, metric.result(), step=step)
                metric.reset_states()

    def log_summaries(self, exploration=True, evaluation=True, critic=True,
            policy=True, forward=True):
        if exploration:
            self.log_metrics(
                "collection",
                "exploration",
                self.n_exploration_episodes
            )
        if evaluation:
            self.log_metrics(
                "collection",
                "evaluation",
                self.n_evaluation_episodes
            )
        if critic:
            self.log_metrics(
                "training",
                "critic",
                self.n_critic_training
            )
        if policy:
            self.log_metrics(
                "training",
                "policy",
                self.n_policy_training
            )
        if forward:
            self.log_metrics(
                "training",
                "forward",
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
        # HINDSIGHT EXPERIENCE (TODO)
        register_change = (
            self._data_buffer["current_goals"][:-1] !=
            self._data_buffer["current_goals"][1:]
        ).any(axis=-1)
        iteration_indices = {
            i: changes.nonzero()[0]
            for i, changes in enumerate(register_change.T) if changes.any()
        }
        for_hindsight = []
        # prediction_error_sum = 0
        # prediction_error_n = 0
        # for simulation, changes in iteration_indices.items():
        #     for iteration in range(self._data_buffer.shape[0]):
        #         next = np.argmax(changes > iteration)
        #         if next or changes[0] > iteration:
        #             for transition_index in changes[next:]:
        #                 state = self._data_buffer["states"][iteration, simulation]
        #                 actual_goal = self._data_buffer["current_goals"][transition_index + 1, simulation]
        #                 prediction = self.agent.get_predictions(
        #                     state[np.newaxis],
        #                     actual_goal[np.newaxis]).numpy()[0]
        #                 prediction = get_time_prediction(
        #                     prediction,              # 10, 4
        #                     actual_goal[np.newaxis]  #  1, 4
        #                 )
        #                 truth = transition_index - iteration
        #                 prediction_error_sum += np.abs(prediction - np.clip(truth, 0, self.prediction_time_window))
        #                 prediction_error_n += 1
        #                 prediction = int(prediction)
        #                 if truth < prediction * (1 - self.pessimism):
        #                     print("YES: sim:{:2d} it:{:2d}  took me {:2d} steps to go from {} to {} at {:2d}, I predicted {:2d} ({:.3f})".format(
        #                         simulation,
        #                         iteration,
        #                         truth,
        #                         self._data_buffer["current_goals"][transition_index, simulation],
        #                         actual_goal,
        #                         transition_index,
        #                         prediction,
        #                         prediction * (1 - self.pessimism)))
        #                     copy = np.copy(self._data_buffer[iteration, simulation])
        #                     copy["goals"] = actual_goal
        #                     for_hindsight.append(copy)
        #                 else:
        #                     print("NO : sim:{:2d} it:{:2d}  took me {:2d} steps to go from {} to {} at {:2d}, I predicted {:2d} ({:.3f})".format(
        #                         simulation,
        #                         iteration,
        #                         truth,
        #                         self._data_buffer["current_goals"][transition_index, simulation],
        #                         actual_goal,
        #                         transition_index,
        #                         prediction,
        #                         prediction * (1 - self.pessimism)))
        # if len(for_hindsight):
        #     hindsight_data = np.vstack(for_hindsight)
        #     self.policy_buffer.integrate(hindsight_data)
        #     n_discoveries = len(hindsight_data)
        # else:
        #     n_discoveries = 0
        self.n_policy_transition_gathered += n_discoveries
        self.n_policy_episodes += self.n_simulations
        time_stop = time.time()
        if exploration:
            tb = self.tb["collection"]["exploration"]
        else:
            tb = self.tb["collection"]["evaluation"]
        #
        n_iterations = self.episode_length * self.n_simulations
        it_per_sec = n_iterations / (time_stop - time_start)
        tb["it_per_sec"](it_per_sec)
        #
        goals = self._data_buffer["goals"]
        current_goals = self._data_buffer["current_goals"]
        goal_reached = (goals == current_goals).all(axis=-1)
        success_rate_percent = 100 * np.mean(goal_reached.any(axis=1))
        tb["success_rate_percent"](success_rate_percent)
        #
        n_uniques = sum([len(np.unique(x, axis=0)) for x in current_goals])
        n_uniques /= self.n_simulations
        tb["diversity_per_ep"](n_uniques)
        #
        distance_at_start = np.sqrt(np.sum(
            (current_goals[:, 0] - goals[:, 0]) ** 2,
            axis=-1)
        )
        distance_at_end = np.sqrt(np.sum(
            (current_goals[:, -1] - goals[:, -1]) ** 2,
            axis=-1)
        )
        delta_distance = np.mean(distance_at_start - distance_at_end)
        tb["delta_distance_to_goal"](delta_distance)
        #
        n_register_change = np.mean(np.sum(register_change, axis=1))
        tb["n_register_change"](n_register_change)
        #
        one_away = np.sum(np.abs(goals - current_goals), axis=-1) == 1
        one_away_successes = np.logical_and(one_away[:-1], goal_reached[1:])
        one_away_fails = np.logical_and(
            np.logical_and(one_away[:-1], np.logical_not(one_away[1:])),
            np.logical_not(goal_reached[1:])
        )
        n_one_away_success = np.sum(one_away_successes)
        n_one_away_fail = np.sum(one_away_fails)
        n_one_away_ends = n_one_away_success + n_one_away_fail
        if n_one_away_success + n_one_away_fail:
            one_away_success_rate = 100 * n_one_away_success / n_one_away_ends
            tb["one_away_sucess_rate"](one_away_success_rate)
        #

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

    def train(self, policy=True, critic=True, forward=True):
        data = self.buffer.sample(self.batch_size)
        losses = self.agent.train(
            data["states"],
            data["goals"],
            data["noisy_actions"],
            policy=policy,
            critic=critic,
            forward=forward,
        )
        tb = self.tb["training"]
        if policy:
            self.n_policy_training += 1
            tb["policy"]["loss"](losses["policy"])
        if critic:
            self.n_critic_training += 1
            tb["critic"]["loss"](losses["critic"])
        if forward:
            self.n_forward_training += 1
            tb["forward"]["loss"](losses["forward"])
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

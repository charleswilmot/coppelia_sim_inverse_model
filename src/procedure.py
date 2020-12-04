import numpy as np
from buffer import Buffer
from agent import Agent
from simulation import SimulationPool, MODEL_PATH
from tensorflow.keras.metrics import Mean
import tensorflow as tf
import time
import os
from visualization import Visualization, visualization_data_type
from collections import OrderedDict
from tensorboard.plugins.hparams import api as hp
from imageio import get_writer
from std_autotune import STDAutoTuner


def get_snr_db(signal, noise, axis=1):
    mean_signal = np.mean(signal, axis=axis, keepdims=True)
    mean_noise = np.mean(noise, axis=axis, keepdims=True)
    std_signal = np.std(signal - mean_signal, axis=axis)
    std_noise = np.std(noise - mean_noise, axis=axis)
    where = std_signal != 0
    if not where.any():
        print("WARNING: signal to noise can't be computed (constant signal), returning NaN")
        return np.nan
    std_signal = std_signal[where]
    std_noise = std_noise[where]
    rms_signal_db = np.log10(std_signal)
    rms_noise_db = np.log10(std_noise)
    return 20 * (rms_signal_db - rms_noise_db)


def compute_critic_target(rewards, critic_estimates, noises, bootstrap, discount_factor, noise_magnitude_limit):
    # all arrays have shape [n_simulation, n_iterations]
    if noise_magnitude_limit == 0:
        lambdo = 1.0
    else:
        lambdo = 1 / noise_magnitude_limit
    targets = np.zeros(shape=rewards.shape, dtype=np.float32)
    targets[:, -1] = rewards[:, -1] + discount_factor * bootstrap
    for i in np.arange(rewards.shape[-1] - 2, -1, -1):
        current_targets = targets[:, i + 1]
        if noise_magnitude_limit == 0:
            alpha = 1.0
        else:
            alpha = 1 - np.exp(- lambdo *
                np.sqrt(np.sum(noises[:, i] ** 2, axis=-1))
            )
        targets[:, i] = rewards[:, i] + discount_factor * (
            alpha * critic_estimates[:, i + 1] +
            (1 - alpha) * current_targets
        )
    return targets


def combine(states, goals):
    return np.concatenate([states, goals.astype(np.float32)], axis=-1)


class Procedure(object):
    def __init__(self, agent_conf, buffer_conf, simulation_conf, procedure_conf):
        #   PROCEDURE CONF
        self.episode_length = procedure_conf.episode_length
        self.updates_per_sample = procedure_conf.updates_per_sample
        self.batch_size = procedure_conf.batch_size
        self.n_simulations = simulation_conf.n
        self.log_freq = procedure_conf.log_freq
        self.movement_mode = procedure_conf.movement_mode
        self.movement_span = int(procedure_conf.movement_span_in_sec / \
                                 procedure_conf.simulation_timestep)
        self.movement_modes = [
            self.movement_mode for i in range(self.n_simulations)
        ]
        self.movement_spans = [
            self.movement_span for i in range(self.n_simulations)
        ]
        self.her_max_replays = procedure_conf.her.max_replays
        self.discount_factor = procedure_conf.discount_factor
        self.noise_magnitude_limit = procedure_conf.noise_magnitude_limit
        self.metabolic_cost_scale = procedure_conf.metabolic_cost_scale
        self.std_autotuner_filter_size = procedure_conf.std_autotuner.filter_size
        self.std_importance = procedure_conf.std_autotuner.std_importance
        self.std_temperature = procedure_conf.std_autotuner.temperature
        self.std_autotuner_plot_path_movement = './std_autotuner_movement/'
        self.std_autotuner_plot_path_primitive = './std_autotuner_primitive/'
        #    HPARAMS
        self._hparams = OrderedDict([
            ("policy_LR", agent_conf.policy_learning_rate),
            ("critic_LR", agent_conf.critic_learning_rate),
            ("buffer", buffer_conf.size),
            ("update_rate", procedure_conf.updates_per_sample),
            ("ep_length", procedure_conf.episode_length),
            ("batch_size", procedure_conf.batch_size),
            ("tau", agent_conf.tau),
            ("target_smoothing", agent_conf.target_smoothing_stddev),
            ("movement_mode", procedure_conf.movement_mode),
            ("movement_span", procedure_conf.movement_span_in_sec),
        ])
        #   OBJECTS
        self.agent = Agent(**agent_conf)
        self.has_movement_primitive = self.agent.has_movement_primitive
        self.buffer = Buffer(**buffer_conf)
        self.movement_std_autotuner = STDAutoTuner(
            procedure_conf.std_autotuner.length,
            self.n_simulations,
            procedure_conf.std_autotuner.min_stddev,
            procedure_conf.std_autotuner.max_stddev,
            importance_ratio=procedure_conf.std_autotuner.importance_ratio
        )
        self.movement_std_autotuner.init(
            np.log(procedure_conf.std_autotuner.stddev_init),
            procedure_conf.std_autotuner.reward_init,
        )
        if self.has_movement_primitive:
            self.primitive_std_autotuner = STDAutoTuner(
                procedure_conf.std_autotuner.length,
                self.n_simulations,
                procedure_conf.std_autotuner.min_stddev,
                procedure_conf.std_autotuner.max_stddev,
                importance_ratio=procedure_conf.std_autotuner.importance_ratio
            )
            self.primitive_std_autotuner.init(
                np.log(procedure_conf.std_autotuner.stddev_init),
                procedure_conf.std_autotuner.reward_init,
            )
        #   SIMULATION POOL
        guis = list(simulation_conf.guis)
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
        self.goal_size = agent_conf.goal_size
        self.state_size = agent_conf.state_size
        self.action_size = int(agent_conf.action_size)
        self.primitive_size = self.agent.primitive_size
        self.n_actions_in_movement = int(procedure_conf.policy_output_size) // self.action_size
        if self.has_movement_primitive:
            self.primitive_discount_factor = self.discount_factor
        self.movement_discount_factor = np.power(self.discount_factor, 1 / self.n_actions_in_movement)

        print("self.goal_size", self.goal_size)
        print("self.state_size", self.state_size)
        print("self.action_size", self.action_size)
        print("self.primitive_size", self.primitive_size)
        print("self.n_actions_in_movement", self.n_actions_in_movement)

        #   DEFINING DATA BUFFERS
        # training
        train_movement_data_type_description = [
            ("states", np.float32, (self.state_size,)),
            ("goals", np.float32, (self.goal_size,)),
            ("critic_targets", np.float32),
            ("noisy_actions", np.float32, (self.action_size,)),
        ]
        train_movement_data_type = np.dtype(train_movement_data_type_description)
        if self.has_movement_primitive:
            train_primitive_data_type_description = [
                ("states", np.float32, (self.state_size,)),
                ("goals", np.float32, (self.goal_size,)),
                ("critic_targets", np.float32),
                ("noisy_actions", np.float32, (self.primitive_size,)),
            ]
            train_primitive_data_type = np.dtype(train_primitive_data_type_description)
            train_data_type = np.dtype([
                ('movement', train_movement_data_type, (self.n_actions_in_movement,)),
                ('primitive', train_primitive_data_type),
            ])
        else:
            train_data_type = np.dtype([
                ('movement', train_movement_data_type, (self.n_actions_in_movement,)),
            ])
        self._train_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=train_data_type
        )
        # logging (complements training)
        log_movement_data_type_description = [
            ("current_goals", np.float32, (self.goal_size,)),
            ("pure_actions", np.float32, (self.action_size,)),
            ("noises", np.float32, (self.action_size,)),
            ("rewards", np.float32),
            ("metabolic_costs", np.float32),
            ("target_return_estimates", np.float32),
        ]
        log_movement_data_type = np.dtype(log_movement_data_type_description)
        if self.has_movement_primitive:
            log_primitive_data_type_description = [
                ("current_goals", np.float32, (self.goal_size,)),
                ("pure_actions", np.float32, (self.primitive_size,)),
                ("noises", np.float32, (self.primitive_size,)),
                ("rewards", np.float32),
                ("metabolic_costs", np.float32),
                ("target_return_estimates", np.float32),
            ]
            log_primitive_data_type = np.dtype(log_primitive_data_type_description)
            log_data_type = np.dtype([
                ('movement', log_movement_data_type, (self.n_actions_in_movement,)),
                ('primitive', log_primitive_data_type_description),
            ])
        else:
            log_data_type = np.dtype([
                ('movement', log_movement_data_type, (self.n_actions_in_movement,)),
            ])
        self._log_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=log_data_type
        )
        # evaluation
        evaluation_movement_data_type_description = [
            ("states", np.float32, (self.state_size,)),
            ("goals", np.float32, (self.goal_size,)),
            ("current_goals", np.float32, (self.goal_size,)),
            ("pure_actions", np.float32, (self.action_size,)),
            ("target_return_estimates", np.float32),
            ("rewards", np.float32),
            ("metabolic_costs", np.float32),
            ("critic_targets", np.float32),
            ("max_step_returns", np.float32),
        ]
        evaluation_movement_data_type = np.dtype(evaluation_movement_data_type_description)
        if self.has_movement_primitive:
            evaluation_primitive_data_type_description = [
                ("states", np.float32, (self.state_size,)),
                ("goals", np.float32, (self.goal_size,)),
                ("current_goals", np.float32, (self.goal_size,)),
                ("pure_actions", np.float32, (self.action_size,)),
                ("target_return_estimates", np.float32),
                ("rewards", np.float32),
                ("metabolic_costs", np.float32),
                ("critic_targets", np.float32),
                ("max_step_returns", np.float32),
            ]
            evaluation_primitive_data_type = np.dtype(evaluation_primitive_data_type_description)
            evaluation_data_type = np.dtype([
                ('movement', evaluation_movement_data_type, (self.n_actions_in_movement,)),
                ('primitive', evaluation_primitive_data_type),
            ])
        else:
            evaluation_data_type = np.dtype([
                ('movement', evaluation_movement_data_type, (self.n_actions_in_movement,)),
            ])
        self._evaluation_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=evaluation_data_type
        )
        # visualization
        self._visualization_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length * self.n_actions_in_movement),
            dtype=visualization_data_type
        )

        # COUNTERS
        self.n_exploration_episodes = 0
        self.n_evaluation_episodes = 0
        self.n_transition_gathered = 0
        self.n_policy_training = 0
        self.n_critic_training = 0
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
        self.tb["collection"]["exploration"]["critic_snr"] = Mean(
            "collection/exploration_critic_snr_db", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["critic_snr"] = Mean(
            "collection/evaluation_critic_snr_db", dtype=tf.float32)
        self.tb["collection"]["exploration"]["metabolic_cost"] = Mean(
            "collection/exploration_metabolic_cost", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["metabolic_cost"] = Mean(
            "collection/evaluation_metabolic_cost", dtype=tf.float32)
        self.tb["collection"]["exploration"]["current_stddev"] = Mean(
            "collection/exploration_current_stddev", dtype=tf.float32)
        #
        self.summary_writer = tf.summary.create_file_writer("logs")
        with self.summary_writer.as_default():
            hp.hparams(self._hparams)
        # TREE STRUCTURE
        os.makedirs('./replays', exist_ok=True)
        os.makedirs('./visualization_data', exist_ok=True)
        os.makedirs(self.std_autotuner_plot_path_movement, exist_ok=True)
        os.makedirs(self.std_autotuner_plot_path_primitive, exist_ok=True)

    def dump_buffers(self):
        os.makedirs('./buffers', exist_ok=True)
        path = "./buffers/buffer_{:6d}.pkl".format(self.n_critic_training)
        self.buffer.dump(path)

    def log_metrics(self, key1, key2, step):
        with self.summary_writer.as_default():
            for name, metric in self.tb[key1][key2].items():
                tf.summary.scalar(metric.name, metric.result(), step=step)
                metric.reset_states()

    def log_summaries(self, exploration=True, evaluation=True, critic=True,
            policy=True):
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
                self.n_exploration_episodes
            )
        if critic:
            self.log_metrics(
                "training",
                "critic",
                self.n_exploration_episodes
            )
        if policy:
            self.log_metrics(
                "training",
                "policy",
                self.n_exploration_episodes
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
        path = "./checkpoints/{:08d}".format(self.n_global_training)
        self.agent.save_weights(path)

    def restore(self, path):
        """Restores the weights from a checkpoint"""
        self.agent.load_weights(path)

    def sample_goals(self, n=None):
        n = self.n_simulations if n is None else n
        """Returns a binary vector corresponding to the goal states of the
        actuators in the simulation for each simulation"""
        return np.random.randint(2, size=(n, self.goal_size))

    def reset_simulations(self, register_states, register_goals, actions=None):
        if actions is None:
            actions = [None] * len(register_states)
        with self.simulation_pool.distribute_args():
            states, current_goals = \
                tuple(zip(*self.simulation_pool.reset(register_states, register_goals, actions)))
        return np.vstack(states), np.vstack(current_goals)

    def replay(self, exploration=False, record=False, n_episodes=10,
            video_name='replay.mp4', resolution=[320, 240]):
        """Applies the current policy in the environment"""
        with self.simulation_pool.specific(0):
            if record:
                writer = get_writer(video_name)
                cam_id = self.simulation_pool.add_camera(
                    position=(1.15, 1.35, 1),
                    orientation=(
                        24 * np.pi / 36,
                        -7 * np.pi / 36,
                         4 * np.pi / 36
                    ),
                    resolution=resolution
                )[0]
            for i in range(n_episodes):
                print("replay: episode", i)
                goals = self.sample_goals(1)
                register_states = self.sample_goals(1)
                states, current_goals = self.reset_simulations(register_states, goals)
                if record:
                    frame = self.simulation_pool.get_frame(cam_id)[0]
                    frame = (frame * 255).astype(np.uint8)
                    for i in range(24):
                        writer.append_data(frame)
                for iteration in range(self.episode_length):
                    if self.has_movement_primitive:
                        pure_primitive, noisy_primitive, noise_primitive, pure_movement, noisy_movement, noise_movement = \
                            self.agent.get_primitive_and_movement(combine(states, goals))
                    else:
                        pure_movement, noisy_movement, noise_movement = self.agent.get_movement(
                            combine(states, goals))
                    if exploration:
                        movement = noisy_movement
                    else:
                        movement = pure_movement
                    if record:
                        states_sequence, current_goals_sequence, metabolic_costs, frames = \
                            self.apply_movement_get_frames(movement, [cam_id])
                        for frame in frames[0]:
                            frame = (frame * 255).astype(np.uint8)
                            writer.append_data(frame)
                    else:
                        states_sequence, current_goals_sequence, metabolic_costs = \
                            self.apply_movement(movement)
                    states = states_sequence[:, -1] # last state of the sequence
            if record:
                writer.close()
                self.simulation_pool.delete_camera(cam_id)

    def replay_overlay(self, max_overlays=5, record=False, n_episodes=10, video_name='overlay.mp4', resolution=[320, 240]):
        """Applies the current policy in the environment"""
        n_overlays = min(max_overlays, self.n_simulations)
        with self.simulation_pool.specific(list(range(n_overlays))):
            if record:
                writer = get_writer(video_name)
                cam_ids = self.simulation_pool.add_camera(
                    position=(1.15, 1.35, 1),
                    orientation=(
                        24 * np.pi / 36,
                        -7 * np.pi / 36,
                         4 * np.pi / 36
                    ),
                    resolution=resolution
                )
            for i in range(n_episodes):
                print("replay: episode", i)
                goals = np.repeat(self.sample_goals(1), n_overlays, axis=0)
                register_states = np.repeat(self.sample_goals(1), n_overlays, axis=0)
                actions = np.random.uniform(size=(n_overlays, 7), low=-1, high=1) * np.linspace(0, 1, n_overlays)[:, np.newaxis]
                states, current_goals = self.reset_simulations(register_states, goals, actions)
                if record:
                    with self.simulation_pool.distribute_args():
                        frame = np.mean(self.simulation_pool.get_frame(cam_ids), axis=0)
                    frame = (frame * 255).astype(np.uint8)
                    for i in range(24):
                        writer.append_data(frame)
                for iteration in range(self.episode_length):
                    if self.has_movement_primitive:
                        _, _, _, movement, _, _ = \
                            self.agent.get_primitive_and_movement(combine(states, goals))
                    else:
                        movement, _, _ = self.agent.get_movement(
                            combine(states, goals))
                    if record:
                        states_sequence, current_goals_sequence, metabolic_costs, frames_per_sim = \
                            self.apply_movement_get_frames(movement, cam_ids)
                        frames = np.mean(frames_per_sim, axis=0)
                        for frame in frames:
                            frame = (frame * 255).astype(np.uint8)
                            writer.append_data(frame)
                    else:
                        states_sequence, current_goals_sequence, metabolic_costs = \
                            self.apply_movement(movement)
                    states = states_sequence[:, -1] # last state of the sequence
            if record:
                writer.close()
                with self.simulation_pool.distribute_args():
                    self.simulation_pool.delete_camera(cam_ids)

    def collect_data(self):
        """Performs one episode of exploration, places data in the buffer"""
        goals = self.sample_goals()
        register_states = self.sample_goals()
        states, current_goals = self.reset_simulations(register_states, goals)
        time_start = time.time()
        for iteration in range(self.episode_length):
            if self.has_movement_primitive:
                pure_primitive, noisy_primitive, noise_primitive, pure_movement, noisy_movement, noise_movement = \
                    self.agent.get_primitive_and_movement(combine(states, goals))
            else:
                pure_movement, noisy_movement, noise_movement = self.agent.get_movement(
                    combine(states, goals))
            states_sequence, current_goals_sequence, metabolic_costs = self.apply_movement(noisy_movement)
            self._train_data_buffer["movement"][:, iteration]["states"] = states_sequence
            self._train_data_buffer["movement"][:, iteration]["noisy_actions"] = noisy_movement
            self._train_data_buffer["movement"][:, iteration]["goals"] = np.repeat(goals[:, np.newaxis], self.n_actions_in_movement, axis=1)
            if self.has_movement_primitive:
                self._train_data_buffer["primitive"][:, iteration]["states"] = states_sequence[:, 0]
                self._train_data_buffer["primitive"][:, iteration]["noisy_actions"] = noisy_primitive
                self._train_data_buffer["primitive"][:, iteration]["goals"] = goals
            # not necessary for training but useful for logging:
            self._log_data_buffer["movement"][:, iteration]["noises"] = noise_movement
            self._log_data_buffer["movement"][:, iteration]["current_goals"] = current_goals_sequence
            self._log_data_buffer["movement"][:, iteration]["pure_actions"] = pure_movement
            self._log_data_buffer["movement"][:, iteration]["metabolic_costs"] = metabolic_costs
            if self.has_movement_primitive:
                self._log_data_buffer["primitive"][:, iteration]["noises"] = noise_primitive
                self._log_data_buffer["primitive"][:, iteration]["current_goals"] = current_goals_sequence[:, 0]
                self._log_data_buffer["primitive"][:, iteration]["pure_actions"] = pure_primitive
                self._log_data_buffer["primitive"][:, iteration]["metabolic_costs"] = np.sum(metabolic_costs, axis=-1)
            states = self.simulation_pool.get_state()
        last_states = states
        goals = self._train_data_buffer["movement"]["goals"]
        current_goals = self._log_data_buffer["movement"]["current_goals"]
        states = self._train_data_buffer["movement"]["states"]
        # COMPUTE CRITIC TARGET
        # MOVEMENT
        distances = np.sum(np.abs(goals - current_goals), axis=-1)
        distances = distances.reshape((self.n_simulations, -1))
        rewards = np.zeros_like(distances, dtype=np.float32)
        rewards[:, :-1] = distances[:, :-1] - distances[:, 1:]
        last_current_goals = self.simulation_pool.get_stateful_objects_states()
        rewards[:, -1] = distances[:, -1] - np.sum(np.abs(last_current_goals - goals[:, -1, -1]), axis=-1)
        self._log_data_buffer["movement"]["rewards"] = \
            rewards.reshape((self.n_simulations, self.episode_length, self.n_actions_in_movement)) - \
            self.metabolic_cost_scale * self._log_data_buffer["movement"]["metabolic_costs"]
        if self.has_movement_primitive:
            pure_target_primitive, noisy_target_primitive, noise_target_primitive, pure_target_movement, noisy_target_movement, noise_target_movement = \
                self.agent.get_primitive_and_movement(combine(states[..., 0, :], goals[..., 0, :]), target=True)  # [..., 0, :] --> need only the first state of the sequence (the actual state)
            _, next_noisy_target_primitive, _, _, next_noisy_target_movement, _ = \
                self.agent.get_primitive_and_movement(combine(last_states, goals[:, -1, -1]), target=True)
            next_noisy_target_movement = next_noisy_target_movement[..., 0, :]   # [..., 0, :] --> need the first target action of the sequence
        else:
            pure_target_movement, noisy_target_movement, noise = self.agent.get_movement(
                combine(states[..., 0, :], goals[..., 0, :]), target=True)  # [..., 0, :] --> need only the first state of the sequence (the actual state)
            pure_target_movement, next_noisy_target_movement, noise = self.agent.get_movement(
                combine(last_states, goals[:, -1, -1]), target=True)
            next_noisy_target_movement = next_noisy_target_movement[..., 0, :]   # [..., 0, :] --> need the first target action of the sequence
        self._log_data_buffer["movement"]["target_return_estimates"] = self.agent.get_movement_return_estimates(
                combine(states, goals), # shape [n_simulations, episode_length, n_actions_in_movement, 25 + 4]
                noisy_target_movement,  # shape [n_simulations, episode_length, n_actions_in_movement, action_size]
                target=True,
        )[..., 0]  # [..., 0] critic's output has shape [..., 1] --> resulting shape [n_simulation, episode_length, n_actions_in_movement]
        movement_bootstrap = self.agent.get_movement_return_estimates(
                combine(last_states, goals[:, -1, -1]),
                next_noisy_target_movement,
                target=True,
        )[..., 0]
        self._train_data_buffer["movement"]["critic_targets"] = compute_critic_target(
            rewards=self._log_data_buffer["movement"]["rewards"].reshape((self.n_simulations, -1)),
            critic_estimates=self._log_data_buffer["movement"]["target_return_estimates"].reshape((self.n_simulations, -1)), # first estimate is not used
            noises=self._log_data_buffer["movement"]["noises"].reshape((self.n_simulations, -1, self.action_size)),
            bootstrap=movement_bootstrap,
            discount_factor=self.movement_discount_factor,
            noise_magnitude_limit=self.noise_magnitude_limit,
        ).reshape((self.n_simulations, self.episode_length, self.n_actions_in_movement))
        # PRIMITIVE
        if self.has_movement_primitive:
            self._log_data_buffer["primitive"]["rewards"] = np.sum(self._log_data_buffer["movement"]["rewards"], axis=-1)
            self._log_data_buffer["primitive"]["target_return_estimates"] = self.agent.get_primitive_return_estimates(
                    combine(self._train_data_buffer["primitive"]["states"], self._train_data_buffer["primitive"]["goals"]), # shape [n_simulations, episode_length, 25 + 4]
                    noisy_target_primitive,  # shape [n_simulations, episode_length, action_size]
                    target=True,
            )[..., 0]  # [..., 0] critic's output has shape [..., 1] --> resulting shape [n_simulation, episode_length]
            primitive_bootstrap = self.agent.get_primitive_return_estimates(
                    combine(last_states, goals[:, -1, -1]),
                    next_noisy_target_primitive,
                    target=True,
            )[..., 0]
            self._train_data_buffer["primitive"]["critic_targets"] = compute_critic_target(
                rewards=self._log_data_buffer["primitive"]["rewards"],
                critic_estimates=self._log_data_buffer["primitive"]["target_return_estimates"], # first estimate is not used
                noises=self._log_data_buffer["primitive"]["noises"],
                bootstrap=primitive_bootstrap,
                discount_factor=self.primitive_discount_factor,
                noise_magnitude_limit=self.noise_magnitude_limit,
            )
        #### COMPUTE CRITIC TARGET DONE!
        movement_log_stddevs = self.agent.get_movement_log_stddevs()
        # rewards = np.sum(self._log_data_buffer["movement"]["rewards"], axis=(1, 2))
        rewards = np.sum(np.abs(self._log_data_buffer["movement"]["target_return_estimates"] - self._train_data_buffer["movement"]["critic_targets"]), axis=(1, 2))
        self.movement_std_autotuner.register_rewards(movement_log_stddevs, rewards)
        movement_log_stddevs = self.movement_std_autotuner.get_log_stddevs(self.std_autotuner_filter_size, self.std_importance, self.std_temperature)
        movement_stddev = np.exp(movement_log_stddevs[len(movement_log_stddevs) // 2])
        self.agent.set_movement_log_stddevs(movement_log_stddevs)
        self.movement_std_autotuner.save_plot(
            self.std_autotuner_plot_path_movement + '{:07d}.png'.format(self.n_exploration_episodes),
            self.std_autotuner_filter_size,
            self.std_importance,
            movement_log_stddevs,
        )
        if self.has_movement_primitive:
            primitive_log_stddevs = self.agent.get_primitive_log_stddevs()
            # rewards = np.sum(self._log_data_buffer["primitive"]["rewards"], axis=(1, 2))
            rewards = np.sum(np.abs(self._log_data_buffer["primitive"]["target_return_estimates"] - self._train_data_buffer["primitive"]["critic_targets"]), axis=-1)
            self.primitive_std_autotuner.register_rewards(primitive_log_stddevs, rewards)
            primitive_log_stddevs = self.primitive_std_autotuner.get_log_stddevs(self.std_autotuner_filter_size, self.std_importance, self.std_temperature)
            primitive_stddev = np.exp(primitive_log_stddevs[len(primitive_log_stddevs) // 2])
            self.agent.set_primitive_log_stddevs(primitive_log_stddevs)
            self.primitive_std_autotuner.save_plot(
                self.std_autotuner_plot_path_primitive + '{:07d}.png'.format(self.n_exploration_episodes),
                self.std_autotuner_filter_size,
                self.std_importance,
                primitive_log_stddevs,
            )
        # HINDSIGHT EXPERIENCE
        for_hindsight = []
        if self.her_max_replays > 0:
            her_goals_per_sim = [
                np.unique(trajectory_goals.reshape((-1, self.goal_size)), axis=0)
                for trajectory_goals in current_goals  # current_goals has shape [n_simulations, episode_length, n_actions_in_movement, goal_size]
            ] # all 'current_goal' visited during one episode, for each sim
            her_goals_per_sim = [
                her_goals[(her_goals != true_goal).any(axis=-1)]
                for her_goals, true_goal in zip(her_goals_per_sim, goals[:, 0, 0])
            ] # filter out the actual true goal pursued during the episode
            her_goals_per_sim = [
                her_goals[-self.her_max_replays:]
                for her_goals in her_goals_per_sim
            ] # keep up to 'her_max_replays' of those goals (from the last)
            movement_rewards_buffer = np.zeros(shape=(self.episode_length * self.n_actions_in_movement), dtype=np.float32)
            for simulation, her_goals in enumerate(her_goals_per_sim):
                metabolic_costs = self.metabolic_cost_scale * self._log_data_buffer["movement"][simulation]["metabolic_costs"]
                for her_goal in her_goals: # for each simulation, for each fake (HER) goal
                    her_data = np.copy(self._train_data_buffer[simulation])
                    her_data["movement"]["goals"] = her_goal
                    goals = her_data["movement"]["goals"]
                    current_goals = self._log_data_buffer["movement"][simulation]["current_goals"]
                    states = her_data["movement"]["states"] # shape [episode_length, n_actions_in_movement, state_size]
                    distances = np.sum(np.abs(goals - current_goals), axis=-1)  # shape [episode_length, n_actions_in_movement]
                    distances = distances.flatten()
                    movement_rewards_buffer[:-1] = distances[:-1] - distances[1:]
                    movement_rewards_buffer[-1] = distances[-1] - np.sum(np.abs(last_current_goals[simulation] - goals[-1, -1]), axis=-1)
                    movement_rewards = movement_rewards_buffer.reshape((self.episode_length, self.n_actions_in_movement)) - metabolic_costs
                    next_state_goal = combine(last_states[simulation], goals[-1, -1])[np.newaxis]
                    if self.has_movement_primitive:
                        pure_target_primitive, noisy_target_primitive, noise_target_primitive, pure_target_movement, noisy_target_movement, noise_target_movement = \
                            self.agent.get_primitive_and_movement(combine(states[..., 0, :], goals[..., 0, :]), target=True)  # [..., 0, :] --> need only the first state of the sequence (the actual state)
                        _, next_noisy_target_primitive, _, _, next_noisy_target_movement, _ = \
                        next_pure_target_primitive, next_noisy_target_primitive, noise, next_pure_target_movement, next_noisy_target_movement, noise = \
                            self.agent.get_primitive_and_movement(next_state_goal, target=True)
                        next_noisy_target_movement = next_noisy_target_movement[..., 0, :]   # [..., 0, :] --> need the first target action of the sequence
                    else:
                        pure_target_movement, noisy_target_movement, noises = self.agent.get_movement(
                            combine(states[..., 0, :], goals[..., 0, :]), target=True) # shape [episode_length, n_actions_in_movement, action_size]
                        next_pure_target_movement, next_noisy_target_movement, next_noise = self.agent.get_movement(
                            next_state_goal, target=True)
                        next_noisy_target_movement = next_noisy_target_movement[..., 0, :]   # [..., 0, :] --> need the first target action of the sequence
                    movement_noises = her_data["movement"]["noisy_actions"] - pure_target_movement.numpy() # shape [episode_length, n_actions_in_movement, action_size]
                    target_return_estimates = self.agent.get_movement_return_estimates(
                            combine(states, goals), # shape [episode_length, n_actions_in_movement, 25 + 4]
                            noisy_target_movement,  # shape [episode_length, n_actions_in_movement, action_size]
                            target=True,
                    )[..., 0].numpy()  # [..., 0] critic's output has shape [..., 1] --> resulting shape [episode_length, n_actions_in_movement]
                    bootstrap = self.agent.get_movement_return_estimates(
                            next_state_goal,
                            next_noisy_target_movement,
                            target=True,
                    )[..., 0].numpy()
                    her_data["movement"]["critic_targets"] = compute_critic_target(
                        rewards=movement_rewards.reshape((1, -1)),
                        critic_estimates=target_return_estimates.reshape((1, -1)), # first estimate is not used
                        noises=movement_noises.reshape((1, -1, self.action_size)),
                        bootstrap=bootstrap,
                        discount_factor=self.movement_discount_factor,
                        noise_magnitude_limit=self.noise_magnitude_limit,
                    ).reshape((self.episode_length, self.n_actions_in_movement))
                    if self.has_movement_primitive:
                        her_data["primitive"] = np.copy(self._train_data_buffer["primitive"][simulation])
                        her_data["primitive"]["goals"] = her_goal
                        primitive_rewards = np.sum(movement_rewards, axis=-1)
                        primitive_noises = her_data["primitive"]["noisy_actions"] - pure_target_primitive.numpy() # shape [episode_length, action_size]
                        target_return_estimates = self.agent.get_primitive_return_estimates(
                                combine(her_data["primitive"]["states"], her_data["primitive"]["goals"]), # shape [episode_length, 25 + 4]
                                noisy_target_primitive,  # shape [episode_length, action_size]
                                target=True,
                        )[..., 0].numpy()  # [..., 0] critic's output has shape [..., 1] --> resulting shape [episode_length]
                        bootstrap = self.agent.get_primitive_return_estimates(
                                next_state_goal,
                                next_noisy_target_primitive,
                                target=True,
                        )[..., 0].numpy()
                        her_data["primitive"]["critic_targets"] = compute_critic_target(
                            rewards=primitive_rewards[np.newaxis],
                            critic_estimates=target_return_estimates[np.newaxis], # first estimate is not used
                            noises=primitive_noises[np.newaxis],
                            bootstrap=bootstrap,
                            discount_factor=self.primitive_discount_factor,
                            noise_magnitude_limit=self.noise_magnitude_limit,
                        ).reshape((self.episode_length,))
                    for_hindsight.append(her_data)
        regular_data = self._train_data_buffer.flatten()
        buffer_data = np.concatenate(for_hindsight + [regular_data], axis=0)
        self.buffer.integrate(buffer_data)
        self.n_transition_gathered += len(buffer_data)
        self.n_exploration_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        self.accumulate_log_data(
            goals=self._train_data_buffer["movement"]["goals"],
            current_goals=self._log_data_buffer["movement"]["current_goals"],
            return_estimates=self._log_data_buffer["movement"]["target_return_estimates"],
            critic_targets=self._train_data_buffer["movement"]["critic_targets"],
            metabolic_costs=self._log_data_buffer["movement"]["metabolic_costs"],
            time=time_stop - time_start,
            exploration=True,
            current_stddev=movement_stddev
        )

    def evaluate(self):
        """Performs one episode of evaluation"""
        goals = self.sample_goals()
        register_states = self.sample_goals()
        states, current_goals = self.reset_simulations(register_states, goals)
        time_start = time.time()
        for iteration in range(self.episode_length):
            if self.has_movement_primitive:
                pure_primitive, noisy_primitive, noise_primitive, pure_movement, noisy_movement, noise_movement = \
                    self.agent.get_primitive_and_movement(combine(states, goals))
            else:
                pure_movement, noisy_movement, noise_movement = self.agent.get_movement(
                    combine(states, goals))
            states_sequence, current_goals_sequence, metabolic_costs = self.apply_movement(pure_movement)
            self._evaluation_data_buffer["movement"][:, iteration]["states"] = states_sequence
            self._evaluation_data_buffer["movement"][:, iteration]["goals"] = np.repeat(goals[:, np.newaxis], self.n_actions_in_movement, axis=1)
            self._evaluation_data_buffer["movement"][:, iteration]["current_goals"] = current_goals_sequence
            self._evaluation_data_buffer["movement"][:, iteration]["pure_actions"] = pure_movement
            self._evaluation_data_buffer["movement"][:, iteration]["metabolic_costs"] = metabolic_costs
            if self.has_movement_primitive:
                self._evaluation_data_buffer["primitive"][:, iteration]["states"] = states_sequence[:, 0]
                self._evaluation_data_buffer["primitive"][:, iteration]["goals"] = goals
                self._evaluation_data_buffer["primitive"][:, iteration]["current_goals"] = current_goals_sequence[:, 0]
                self._evaluation_data_buffer["primitive"][:, iteration]["pure_actions"] = pure_primitive
                self._evaluation_data_buffer["primitive"][:, iteration]["metabolic_costs"] = np.sum(metabolic_costs, axis=-1)
            states = self.simulation_pool.get_state()
        last_states = states
        states = self._evaluation_data_buffer["movement"]["states"]
        pure_movement = self._evaluation_data_buffer["movement"]["pure_actions"]
        goals = self._evaluation_data_buffer["movement"]["goals"]
        current_goals = self._evaluation_data_buffer["movement"]["current_goals"]
        # COMPUTE CRITIC TARGET
        # MOVEMENT
        distances = np.sum(np.abs(goals - current_goals), axis=-1)
        distances = distances.reshape((self.n_simulations, -1))
        rewards = np.zeros_like(distances, dtype=np.float32)
        rewards[:, :-1] = distances[:, :-1] - distances[:, 1:]
        last_current_goals = self.simulation_pool.get_stateful_objects_states()
        rewards[:, -1] = distances[:, -1] - np.sum(np.abs(last_current_goals - goals[:, -1, -1]), axis=-1)
        self._evaluation_data_buffer["movement"]["rewards"] = \
            rewards.reshape((self.n_simulations, self.episode_length, self.n_actions_in_movement)) - \
            self.metabolic_cost_scale * self._evaluation_data_buffer["movement"]["metabolic_costs"]
        if self.has_movement_primitive:
            pure_target_primitive, noisy_target_primitive, noise_target_primitive, pure_target_movement, noisy_target_movement, noise_target_movement = \
                self.agent.get_primitive_and_movement(combine(states[..., 0, :], goals[..., 0, :]), target=True)  # [..., 0, :] --> need only the first state of the sequence (the actual state)
            _, next_noisy_target_primitive, _, _, next_noisy_target_movement, _ = \
            next_pure_target_primitive, next_noisy_target_primitive, noise, next_pure_target_movement, next_noisy_target_movement, noise = \
                self.agent.get_primitive_and_movement(combine(last_states, goals[:, -1, -1]), target=True)
            next_noisy_target_movement = next_noisy_target_movement[..., 0, :]   # [..., 0, :] --> need the first target action of the sequence
        else:
            pure_target_movement, noisy_target_movement, noise = self.agent.get_movement(
                combine(states[..., 0, :], goals[..., 0, :]), target=True)  # [..., 0, :] --> need only the first state of the sequence (the actual state)
            pure_target_movement, next_noisy_target_movement, noise = self.agent.get_movement(
                combine(last_states, goals[:, -1, -1]), target=True)
            next_noisy_target_movement = next_noisy_target_movement[..., 0, :]   # [..., 0, :] --> need the first target action of the sequence
        self._evaluation_data_buffer["movement"]["target_return_estimates"] = self.agent.get_movement_return_estimates(
                combine(states, goals), # shape [n_simulations, episode_length, n_actions_in_movement, 25 + 4]
                noisy_target_movement,  # shape [n_simulations, episode_length, n_actions_in_movement, action_size]
                target=True,
        )[..., 0]  # [..., 0] critic's output has shape [..., 1] --> resulting shape [n_simulation, episode_length, n_actions_in_movement]
        movement_bootstrap = self.agent.get_movement_return_estimates(
                combine(last_states, goals[:, -1, -1]),
                next_noisy_target_movement,
                target=True,
        )[..., 0]
        self._train_data_buffer["movement"]["critic_targets"] = compute_critic_target(
            rewards=self._evaluation_data_buffer["movement"]["rewards"].reshape((self.n_simulations, -1)),
            critic_estimates=self._evaluation_data_buffer["movement"]["target_return_estimates"].reshape((self.n_simulations, -1)), # first estimate is not used
            noises=np.zeros_like(self._evaluation_data_buffer["movement"]["pure_actions"]).reshape((self.n_simulations, -1, self.action_size)),
            bootstrap=movement_bootstrap,
            discount_factor=self.movement_discount_factor,
            noise_magnitude_limit=self.noise_magnitude_limit,
        ).reshape((self.n_simulations, self.episode_length, self.n_actions_in_movement))
        # PRIMITIVE
        if self.has_movement_primitive:
            self._evaluation_data_buffer["primitive"]["rewards"] = np.sum(self._evaluation_data_buffer["movement"]["rewards"], axis=-1)
            self._evaluation_data_buffer["primitive"]["target_return_estimates"] = self.agent.get_primitive_return_estimates(
                    combine(self._evaluation_data_buffer["primitive"]["states"], self._evaluation_data_buffer["primitive"]["goals"]), # shape [n_simulations, episode_length, 25 + 4]
                    noisy_target_primitive,  # shape [n_simulations, episode_length, action_size]
                    target=True,
            )[..., 0]  # [..., 0] critic's output has shape [..., 1] --> resulting shape [n_simulation, episode_length, n_actions_in_movement]
            primitive_bootstrap = self.agent.get_primitive_return_estimates(
                    combine(last_states, goals[:, -1, -1]),
                    next_noisy_target_primitive,
                    target=True,
            )[..., 0]
            self._train_data_buffer["primitive"]["critic_targets"] = compute_critic_target(
                rewards=self._evaluation_data_buffer["primitive"]["rewards"],
                critic_estimates=self._evaluation_data_buffer["primitive"]["target_return_estimates"], # first estimate is not used
                noises=np.zeros_like(self._evaluation_data_buffer["primitive"]["pure_actions"]),
                bootstrap=primitive_bootstrap,
                discount_factor=self.primitive_discount_factor,
                noise_magnitude_limit=self.noise_magnitude_limit,
            )
        #### COMPUTE CRITIC TARGET DONE!
        #### COMPUTE MAX STEP RETURN
        # movement max_step return
        prev = movement_bootstrap
        for it in np.arange(self.episode_length - 1, -1, -1):
            for sub_it in np.arange(self.n_actions_in_movement - 1, -1, -1):
                self._evaluation_data_buffer["movement"][:, it, sub_it]["max_step_returns"] = \
                    self.movement_discount_factor * prev + self._evaluation_data_buffer["movement"][:, it, sub_it]["rewards"]
                prev = self._evaluation_data_buffer["movement"][:, it, sub_it]["max_step_returns"]
        # primitive max_step return
        if self.has_movement_primitive:
            prev = primitive_bootstrap
            for it in np.arange(self.episode_length - 1, -1, -1):
                self._evaluation_data_buffer["primitive"][:, it]["max_step_returns"] = \
                    self.primitive_discount_factor * prev + self._evaluation_data_buffer["primitive"][:, it]["rewards"]
                prev = self._evaluation_data_buffer["primitive"][:, it]["max_step_returns"]
        #### COMPUTE MAX STEP RETURN DONE!
        self.n_evaluation_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        self.accumulate_log_data(
            goals=self._evaluation_data_buffer["movement"]["goals"],
            current_goals=self._evaluation_data_buffer["movement"]["current_goals"],
            return_estimates=self._evaluation_data_buffer["movement"]["target_return_estimates"],
            critic_targets=self._evaluation_data_buffer["movement"]["max_step_returns"],
            metabolic_costs=self._evaluation_data_buffer["movement"]["metabolic_costs"],
            time=time_stop - time_start,
            exploration=False,
        )
        # LOG DATA FOR CUSTOM VISUALIZATION
        self._visualization_data_buffer["rewards"] = self._evaluation_data_buffer["movement"]["rewards"].reshape((self.n_simulations, self.episode_length * self.n_actions_in_movement))
        self._visualization_data_buffer["target_return_estimates"] = self._evaluation_data_buffer["movement"]["target_return_estimates"].reshape((self.n_simulations, self.episode_length * self.n_actions_in_movement))
        self._visualization_data_buffer["critic_targets"] = self._evaluation_data_buffer["movement"]["critic_targets"].reshape((self.n_simulations, self.episode_length * self.n_actions_in_movement))
        self._visualization_data_buffer["max_step_returns"] = self._evaluation_data_buffer["movement"]["max_step_returns"].reshape((self.n_simulations, self.episode_length * self.n_actions_in_movement))
        with open("./visualization_data/{}_critic.dat".format(self.episode_length * self.n_actions_in_movement), 'ab') as f:
            f.write(self._visualization_data_buffer.tobytes())

    def accumulate_log_data(self, goals, current_goals, return_estimates,
            critic_targets, metabolic_costs, time, exploration, current_stddev=None):
        if exploration:
            tb = self.tb["collection"]["exploration"]
        else:
            tb = self.tb["collection"]["evaluation"]
        #
        n_iterations = self.episode_length * self.n_simulations
        it_per_sec = n_iterations / time
        tb["it_per_sec"](it_per_sec)
        #
        goal_reached = (goals == current_goals).all(axis=-1)
        success_rate_percent = 100 * np.mean(goal_reached.any(axis=(1, 2)))
        tb["success_rate_percent"](success_rate_percent)
        #
        n_uniques = sum([len(np.unique(x.reshape((-1, self.goal_size)), axis=0)) for x in current_goals])
        n_uniques /= self.n_simulations
        tb["diversity_per_ep"](n_uniques)
        #
        distance_at_start = np.sqrt(np.sum(
            (current_goals[:, 0, 0] - goals[:, 0, 0]) ** 2,
            axis=-1)
        )
        distance_at_end = np.sqrt(np.sum(
            (current_goals[:, -1, -1] - goals[:, -1, -1]) ** 2,
            axis=-1)
        )
        delta_distance = np.mean(distance_at_start - distance_at_end)
        tb["delta_distance_to_goal"](delta_distance)
        #
        flattend_current_goals = current_goals.reshape((self.n_simulations, self.episode_length * self.n_actions_in_movement, -1))
        register_change = (
            flattend_current_goals[:, :-1] != flattend_current_goals[:, 1:]
        ).any(axis=-1)
        n_register_change = np.mean(np.sum(register_change, axis=1))
        tb["n_register_change"](n_register_change)
        #
        one_away = np.sum(np.abs(goals - current_goals), axis=-1) == 1
        one_away = one_away.reshape((self.n_simulations, self.n_actions_in_movement * self.episode_length))
        goal_reached = goal_reached.reshape((self.n_simulations, self.n_actions_in_movement * self.episode_length))
        one_away_successes = np.logical_and(one_away[:-1], goal_reached[1:])
        one_away_fails = np.logical_and(
            np.logical_and(one_away[:-1], np.logical_not(one_away[1:])),
            np.logical_not(goal_reached[1:])
        )
        n_one_away_success = np.sum(one_away_successes)
        n_one_away_fail = np.sum(one_away_fails)
        n_one_away_ends = n_one_away_success + n_one_away_fail
        if n_one_away_ends:
            one_away_success_rate = 100 * n_one_away_success / n_one_away_ends
            tb["one_away_sucess_rate"](one_away_success_rate)
        #
        signal = critic_targets.reshape((self.n_simulations, self.n_actions_in_movement * self.episode_length))
        noise = (critic_targets - return_estimates).reshape((self.n_simulations, self.n_actions_in_movement * self.episode_length))
        critic_snr = get_snr_db(signal, noise)
        tb["critic_snr"](np.mean(critic_snr))
        #
        tb["metabolic_cost"](np.mean(metabolic_costs))
        #
        if exploration:
            tb["current_stddev"](current_stddev)

    def get_data(self):
        states, current_goals = tuple(zip(*self.simulation_pool.get_data()))
        return np.vstack(states), np.vstack(current_goals)

    def apply_movement(self, movement):
        # movement must have shape [n_simulations, n_actions_in_movement, action_size]
        with self.simulation_pool.distribute_args():
            states, current_goals, metabolic_costs = \
                tuple(zip(*self.simulation_pool.apply_movement(
                    movement,
                    mode=self.movement_modes,
                    span=self.movement_spans
                )))
        return np.stack(states), np.stack(current_goals), np.array(metabolic_costs)

    def apply_movement_get_frames(self, movement, cam_ids):
        # movement must have shape [n_simulations, n_actions_in_movement, action_size]
        with self.simulation_pool.distribute_args():
            states, current_goals, metabolic_costs, frames = \
                tuple(zip(*self.simulation_pool.apply_movement_get_frames(
                    movement,
                    cam_ids,
                    mode=self.movement_modes,
                    span=self.movement_spans
                )))
        return np.array(states), np.array(current_goals), np.array(metabolic_costs), np.array(frames)

    def train(self, policy=True, critic=True):
        data = self.buffer.sample(self.batch_size)
        if self.has_movement_primitive:
            movement_critic_states = combine(data["movement"]["states"], data["movement"]["goals"]) # shape [batch_size, n_actions_in_movement, 25 + 4]
            movement_policy_states = data["primitive"]["noisy_actions"] # shape [batch_size, primitive_size]
            movement_losses = self.agent.train_movement(
                movement_policy_states, # shape [batch_size, primitive_size]
                movement_critic_states, # shape [batch_size, n_actions_in_movement, 25 + 4]
                data["movement"]["noisy_actions"], # shape [batch_size, n_actions_in_movement, 7]
                data["movement"]["critic_targets"], # shape [batch_size, n_actions_in_movement]
                policy=policy,
                critic=critic,
            )
            primitive_policy_states = combine(data["primitive"]["states"], data["primitive"]["goals"]) # shape [batch_size, 25 + 4]
            primitive_critic_states = primitive_policy_states.reshape((-1, 1, self.state_size + self.goal_size))
            primitive_losses = self.agent.train_primitive(
                primitive_policy_states, # shape [batch_size, 25 + 4]
                primitive_critic_states, # shape [batch_size, 1, 25 + 4]
                data["primitive"]["noisy_actions"].reshape((-1, 1, self.primitive_size)), # shape [batch_size, 1, primitive_size]
                data["primitive"]["critic_targets"].reshape((-1, 1)), # shape [batch_size, 1]
                policy=policy,
                critic=critic,
            )
        else:
            movement_critic_states = combine(data["movement"]["states"], data["movement"]["goals"]) # shape [batch_size, n_actions_in_movement, 25 + 4]
            movement_policy_states = movement_critic_states[:, 0] # shape [batch_size, 25 + 4]
            movement_losses = self.agent.train_movement(
                movement_policy_states, # shape [batch_size, 25 + 4]
                movement_critic_states, # shape [batch_size, n_actions_in_movement, 25 + 4]
                data["movement"]["noisy_actions"], # shape [batch_size, n_actions_in_movement, 7]
                data["movement"]["critic_targets"], # shape [batch_size, n_actions_in_movement]
                policy=policy,
                critic=critic,
            )
        tb = self.tb["training"]
        if policy:
            self.n_policy_training += 1
            tb["policy"]["loss"](movement_losses["policy"])
        if critic:
            self.n_critic_training += 1
            tb["critic"]["loss"](movement_losses["critic"])
        self.n_global_training += 1
        return movement_losses

    def collect_and_train(self, policy=True, critic=True):
        self.collect_data()
        while self.current_training_ratio < self.updates_per_sample:
            self.train(policy=policy, critic=critic)

    def collect_train_and_log(self, policy=True, critic=True, evaluation=False):
        self.collect_and_train(policy=policy, critic=critic)
        if evaluation:
            self.evaluate()
        if self.n_global_training % self.log_freq == 0:
            self.log_summaries(exploration=True, evaluation=evaluation,
                policy=policy, critic=critic)

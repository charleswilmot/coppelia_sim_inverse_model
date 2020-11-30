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


def compute_critic_target(rewards, critic_estimates, noises, discount_factor, noise_magnitude_limit, one_step=False):
    if noise_magnitude_limit == 0:
        lambdo = 1.0
    else:
        lambdo = 1 / noise_magnitude_limit
    shape = np.copy(rewards.shape)
    shape[-1] -= 1
    targets = np.zeros(shape=shape, dtype=np.float32)
    current_targets = critic_estimates[:, -1]
    for i in np.arange(shape[-1] - 1, -1, -1):
        if one_step or noise_magnitude_limit == 0:
            alpha = 1.0
        else:
            alpha = 1 - np.exp(- lambdo *
                np.sqrt(np.sum(noises[:, i] ** 2, axis=-1))
            )
        targets[:, i] = rewards[:, i] + discount_factor * (
            alpha * critic_estimates[:, i + 1] +
            (1 - alpha) * current_targets
        )
        current_targets = targets[:, i]
    return targets


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
        self.one_step = procedure_conf.one_step
        self.metabolic_cost_scale = procedure_conf.metabolic_cost_scale
        self.std_autotuner = STDAutoTuner(
            procedure_conf.std_autotuner.length,
            self.n_simulations,
            procedure_conf.std_autotuner.min_stddev,
            procedure_conf.std_autotuner.max_stddev,
            importance_ratio=procedure_conf.std_autotuner.importance_ratio
        )
        self.std_autotuner.init(
            np.log(procedure_conf.std_autotuner.stddev_init),
            procedure_conf.std_autotuner.reward_init,
        )
        self.std_autotuner_filter_size = procedure_conf.std_autotuner.filter_size
        self.std_importance = procedure_conf.std_autotuner.std_importance
        self.std_temperature = procedure_conf.std_autotuner.temperature
        self.std_autotuner_plot_path = './std_autotuner/'
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
        self.buffer = Buffer(**buffer_conf)
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
        self.action_size = agent_conf.action_size

        print("self.goal_size", self.goal_size)
        print("self.state_size", self.state_size)
        print("self.action_size", self.action_size)

        #   DEFINING DATA BUFFERS
        # training
        self._train_data_type = np.dtype([
            ("states", np.float32, self.state_size),
            ("noisy_actions", np.float32, self.action_size),
            ("goals", np.float32, self.goal_size),
            ("critic_targets", np.float32),
        ])
        self._train_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=self._train_data_type
        )
        # logging (complements training)
        self._log_data_type = np.dtype([
            ("current_goals", np.float32, self.goal_size),
            ("pure_actions", np.float32, self.action_size),
            ("noises", np.float32, self.action_size),
            ("rewards", np.float32),
            ("metabolic_costs", np.float32),
            ("target_return_estimates", np.float32),
        ])
        self._log_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=self._log_data_type
        )
        # evaluation
        self._evaluation_data_type = np.dtype([
            ("states", np.float32, self.state_size),
            ("goals", np.float32, self.goal_size),
            ("current_goals", np.float32, self.goal_size),
            ("pure_actions", np.float32, self.action_size),
            ("return_estimates", np.float32),
            ("rewards", np.float32),
            ("metabolic_costs", np.float32),
            ("critic_targets", np.float32),
            ("max_step_returns", np.float32),
        ])
        self._evaluation_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=self._evaluation_data_type
        )
        # visualization
        self._visualization_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
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
        os.makedirs(self.std_autotuner_plot_path, exist_ok=True)

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
                    pure_actions, noisy_actions, noises = self.agent.get_actions(
                        states, goals)
                    if exploration:
                        actions = noisy_actions
                    else:
                        actions = pure_actions
                    if record:
                        states, current_goals, metabolic_costs, frames = \
                            self.apply_action_get_frames(actions, [cam_id])
                        for frame in frames[0]:
                            frame = (frame * 255).astype(np.uint8)
                            writer.append_data(frame)
                    else:
                        states, current_goals, metabolic_costs = \
                            self.apply_action(actions)
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
                    actions, _, __ = self.agent.get_actions(states, goals)
                    if record:
                        states, current_goals, metabolic_costs, frames_per_sim = \
                            self.apply_action_get_frames(actions, cam_ids)
                        frames = np.mean(frames_per_sim, axis=0)
                        for frame in frames:
                            frame = (frame * 255).astype(np.uint8)
                            writer.append_data(frame)
                    else:
                        states, current_goals, metabolic_costs = \
                            self.apply_action(actions)
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
            pure_actions, noisy_actions, noises = self.agent.get_actions(
                states, goals)
            self._train_data_buffer[:, iteration]["states"] = states
            self._train_data_buffer[:, iteration]["noisy_actions"] = noisy_actions
            self._train_data_buffer[:, iteration]["goals"] = goals
            # not necessary for training but useful for logging:
            self._log_data_buffer[:, iteration]["noises"] = noises
            self._log_data_buffer[:, iteration]["current_goals"] = current_goals
            self._log_data_buffer[:, iteration]["pure_actions"] = pure_actions
            states, current_goals, metabolic_costs = self.apply_action(noisy_actions)
            self._log_data_buffer[:, iteration]["metabolic_costs"] = metabolic_costs
        goals = self._train_data_buffer["goals"]
        current_goals = self._log_data_buffer["current_goals"]
        # COMPUTE TARGETS
        states = self._train_data_buffer["states"]
        # critic target (valid until :-1)
        distances = np.sum(np.abs(goals - current_goals), axis=-1)
        self._log_data_buffer[:, :-1]["rewards"] = \
            distances[:, :-1] - distances[:, 1:] - \
            self.metabolic_cost_scale * self._log_data_buffer[:, :-1]["metabolic_costs"]
        pure_target_actions, noisy_target_actions, noise = self.agent.get_actions(
            states, goals, target=True)
        self._log_data_buffer["target_return_estimates"] = self.agent.get_return_estimates(
                states,
                noisy_target_actions,
                goals,
                target=True,
        )[..., 0]
        self._train_data_buffer[:, :-1]["critic_targets"] = compute_critic_target(
            rewards=self._log_data_buffer["rewards"],  # last reward is not used
            critic_estimates=self._log_data_buffer["target_return_estimates"], # first estimate is not used
            noises=self._log_data_buffer["noises"],  # last noise is not used
            discount_factor=self.discount_factor,
            noise_magnitude_limit=self.noise_magnitude_limit,
            one_step=self.one_step,
        )
        log_stddevs = self.agent.get_log_stddevs()
        # rewards = np.sum(self._log_data_buffer["rewards"], axis=-1)
        rewards = np.sum(np.abs(self._log_data_buffer[:, :-1]["target_return_estimates"] - self._train_data_buffer[:, :-1]["critic_targets"]), axis=-1)
        self.std_autotuner.register_rewards(log_stddevs, rewards)
        log_stddevs = self.std_autotuner.get_log_stddevs(self.std_autotuner_filter_size, self.std_importance, self.std_temperature)
        stddev = np.exp(log_stddevs[len(log_stddevs) // 2])
        self.agent.set_log_stddevs(log_stddevs)
        self.std_autotuner.save_plot(
            self.std_autotuner_plot_path + '{:07d}.png'.format(self.n_exploration_episodes),
            self.std_autotuner_filter_size,
            self.std_importance,
            log_stddevs,
        )
        # HINDSIGHT EXPERIENCE
        for_hindsight = []
        if self.her_max_replays > 0:
            her_goals_per_sim = [
                np.unique(trajectory_goals, axis=0)
                for trajectory_goals in current_goals
            ] # all 'current_goal' visited during one episode, for each sim
            her_goals_per_sim = [
                her_goals[(her_goals != true_goal).any(axis=-1)]
                for her_goals, true_goal in zip(her_goals_per_sim, goals[:, 0])
            ] # filter out the actual true goal pursued during the episode
            her_goals_per_sim = [
                her_goals[-self.her_max_replays:]
                for her_goals in her_goals_per_sim
            ] # keep up to 'her_max_replays' of those goals (from the last)
            rewards_buffer = np.zeros(shape=self.episode_length, dtype=np.float32)
            target_return_estimates_buffer = np.zeros(shape=self.episode_length, dtype=np.float32)
            for simulation, her_goals in enumerate(her_goals_per_sim):
                metabolic_costs = self.metabolic_cost_scale * self._log_data_buffer[simulation, :-1]["metabolic_costs"]
                for her_goal in her_goals: # for each simulation, for each fake (HER) goal
                    her_data = np.copy(self._train_data_buffer[simulation])
                    her_data["goals"] = her_goal
                    goals = her_data["goals"]
                    current_goals = self._log_data_buffer[simulation]["current_goals"]
                    states = her_data["states"]
                    distances = np.sum(np.abs(goals - current_goals), axis=-1)
                    rewards_buffer[:-1] = distances[:-1] - distances[1:] - metabolic_costs
                    pure_target_actions, noisy_target_actions, noises = self.agent.get_actions(
                        states, goals, target=True)
                    noises_buffer = her_data["noisy_actions"] - pure_target_actions
                    her_data[:-1]["critic_targets"] = compute_critic_target(
                        rewards=rewards_buffer[np.newaxis],  # last reward is not used
                        critic_estimates=target_return_estimates_buffer[np.newaxis], # first estimate is not used
                        noises=noises_buffer[np.newaxis],  # last noise is not used
                        discount_factor=self.discount_factor,
                        noise_magnitude_limit=self.noise_magnitude_limit,
                        one_step=self.one_step,
                    )
                    for_hindsight.append(her_data[:-1])
        regular_data = self._train_data_buffer[:, :-1].flatten()
        buffer_data = np.concatenate(for_hindsight + [regular_data], axis=0)
        self.buffer.integrate(buffer_data)
        self.n_transition_gathered += len(buffer_data)
        self.n_exploration_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        self.accumulate_log_data(
            goals=self._train_data_buffer["goals"],
            current_goals=self._log_data_buffer["current_goals"],
            return_estimates=self._log_data_buffer[:, 1:-1]["target_return_estimates"],
            critic_targets=self._train_data_buffer[:, 1:-1]["critic_targets"],
            metabolic_costs=self._log_data_buffer["metabolic_costs"],
            time=time_stop - time_start,
            exploration=True,
            current_stddev=stddev
        )

    def evaluate(self):
        """Performs one episode of evaluation"""
        goals = self.sample_goals()
        register_states = self.sample_goals()
        states, current_goals = self.reset_simulations(register_states, goals)
        time_start = time.time()
        for iteration in range(self.episode_length):
            pure_actions, noisy_actions, noise = self.agent.get_actions(
                states, goals)
            self._evaluation_data_buffer[:, iteration]["states"] = states
            self._evaluation_data_buffer[:, iteration]["goals"] = goals
            self._evaluation_data_buffer[:, iteration]["current_goals"] = current_goals
            self._evaluation_data_buffer[:, iteration]["pure_actions"] = pure_actions
            states, current_goals, metabolic_costs = self.apply_action(pure_actions)
            self._evaluation_data_buffer[:, iteration]["metabolic_costs"] = metabolic_costs
        states = self._evaluation_data_buffer["states"]
        pure_actions = self._evaluation_data_buffer["pure_actions"]
        goals = self._evaluation_data_buffer["goals"]
        current_goals = self._evaluation_data_buffer["current_goals"]
        # BATCH PROCESSING
        return_estimates = self.agent.get_return_estimates(
            states, pure_actions, goals)[..., 0]
        self._evaluation_data_buffer["return_estimates"] = \
            return_estimates
        # COMPUTE TARGETS
        states = self._evaluation_data_buffer["states"]
        # critic target (valid until :-1)
        distances = np.sum(np.abs(goals - current_goals), axis=-1)
        self._evaluation_data_buffer[:, :-1]["rewards"] = \
            distances[:, :-1] - distances[:, 1:] - \
            self.metabolic_cost_scale * self._evaluation_data_buffer[:, :-1]["metabolic_costs"]
        self._evaluation_data_buffer[:, :-1]["critic_targets"] = compute_critic_target(
            rewards=self._evaluation_data_buffer["rewards"],  # last reward is not used
            critic_estimates=self._evaluation_data_buffer["return_estimates"], # first estimate is not used
            noises=np.zeros_like(self._evaluation_data_buffer["rewards"]),  # last noise is not used
            discount_factor=self.discount_factor,
            noise_magnitude_limit=self.noise_magnitude_limit,
            one_step=self.one_step,
        )
        prev = self._evaluation_data_buffer[:, -1]["return_estimates"]
        self._evaluation_data_buffer[:, -1]["max_step_returns"] = prev
        for it in np.arange(self.episode_length - 2, -1, -1):
            self._evaluation_data_buffer[:, it]["max_step_returns"] = \
                self.discount_factor * prev + self._evaluation_data_buffer[:, it]["rewards"]
            prev = self._evaluation_data_buffer[:, it]["max_step_returns"]
        self.n_evaluation_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        self.accumulate_log_data(
            goals=self._evaluation_data_buffer["goals"],
            current_goals=self._evaluation_data_buffer["current_goals"],
            return_estimates=self._evaluation_data_buffer[:, :-1]["return_estimates"],
            critic_targets=self._evaluation_data_buffer[:, :-1]["max_step_returns"],
            metabolic_costs=self._evaluation_data_buffer["metabolic_costs"],
            time=time_stop - time_start,
            exploration=False,
        )
        # LOG DATA FOR CUSTOM VISUALIZATION
        self._visualization_data_buffer["rewards"] = self._evaluation_data_buffer["rewards"]
        self._visualization_data_buffer["return_estimates"] = self._evaluation_data_buffer["return_estimates"]
        self._visualization_data_buffer["critic_targets"] = self._evaluation_data_buffer["critic_targets"]
        self._visualization_data_buffer["max_step_returns"] = self._evaluation_data_buffer["max_step_returns"]
        with open("./visualization_data/{}_critic.dat".format(self.episode_length), 'ab') as f:
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
        register_change = (
            current_goals[:, :-1] != current_goals[:, 1:]
        ).any(axis=-1)
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
        signal = critic_targets
        noise = critic_targets - return_estimates
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

    def apply_action(self, actions):
        with self.simulation_pool.distribute_args():
            states, current_goals, metabolic_costs = \
                tuple(zip(*self.simulation_pool.apply_movement(
                    actions,
                    mode=self.movement_modes,
                    span=self.movement_spans
                )))
        return np.vstack(states), np.vstack(current_goals), np.array(metabolic_costs)

    def apply_action_get_frames(self, actions, cam_ids):
        with self.simulation_pool.distribute_args():
            states, current_goals, metabolic_costs, frames = \
                tuple(zip(*self.simulation_pool.apply_movement_get_frames(
                    actions,
                    cam_ids,
                    mode=self.movement_modes,
                    span=self.movement_spans
                )))
        return np.array(states), np.array(current_goals), np.array(metabolic_costs), np.array(frames)

    def train(self, policy=True, critic=True):
        data = self.buffer.sample(self.batch_size)
        losses = self.agent.train(
            data["states"],
            data["noisy_actions"],
            data["goals"],
            data["critic_targets"],
            policy=policy,
            critic=critic,
        )
        tb = self.tb["training"]
        if policy:
            self.n_policy_training += 1
            tb["policy"]["loss"](losses["policy"])
        if critic:
            self.n_critic_training += 1
            tb["critic"]["loss"](losses["critic"])
        self.n_global_training += 1
        return losses

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

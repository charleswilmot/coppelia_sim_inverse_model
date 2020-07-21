import numpy as np
from buffer import Buffer
from agent import Agent
from simulation import SimulationPool, MODEL_PATH
from tensorflow.keras.metrics import Mean
import tensorflow as tf
import time
from visualization import Visualization
from collections import OrderedDict
from tensorboard.plugins.hparams import api as hp
from imageio import get_writer


class Procedure(object):
    def __init__(self, agent_conf, policy_buffer_conf, critic_buffer_conf,
            simulation_conf, procedure_conf):
        #   PROCEDURE CONF
        self.episode_length = procedure_conf.episode_length
        self.critic_updates_per_sample = procedure_conf.critic_updates_per_sample
        self.policy_updates_per_sample = procedure_conf.policy_updates_per_sample
        self.batch_size = procedure_conf.batch_size
        self.n_simulations = simulation_conf.n
        self.prediction_filter_lookup = procedure_conf.prediction_filter_lookup
        self.prediction_filter = np.array([1] + [
            -1 / self.prediction_filter_lookup
            for i in range(self.prediction_filter_lookup)
        ], dtype=np.float32)
        #    HPARAMS
        self._hparams = OrderedDict([
            ("policy_LR", agent_conf.policy_learning_rate),
            ("critic_LR", agent_conf.critic_learning_rate),
            ("policy_buffer", policy_buffer_conf.size),
            ("critic_buffer", critic_buffer_conf.size),
            ("policy_update_rate", procedure_conf.policy_updates_per_sample),
            ("critic_update_rate", procedure_conf.critic_updates_per_sample),
            ("pred_lookup", agent_conf.prediction_time_window),
            ("ep_length", procedure_conf.episode_length),
            ("batch_size", procedure_conf.batch_size),
            ("noise_type", agent_conf.exploration.type),
            ("noise_std", agent_conf.exploration.stddev),
            ("noise_damp", agent_conf.exploration.damping),
            ("movement_mode", procedure_conf.movement_mode),
            ("movement_span", procedure_conf.movement_span_in_sec),
            ("her_lookup", procedure_conf.her_lookup),
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
        self.policy_buffer = Buffer(**policy_buffer_conf)
        self.critic_buffer = Buffer(**critic_buffer_conf)
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
            self.prediction_size = agent_conf.prediction_time_window

        print("self.goal_size", self.goal_size)
        print("self.state_size", self.state_size)
        print("self.action_size", self.action_size)
        print("self.prediction_size", self.prediction_size)

        #   DEFINING DATA BUFFERS
        # policy
        self._policy_data_type = np.dtype([
            ("states", np.float32, self.state_size),
            ("goals", np.float32, self.goal_size),
            ("current_goals", np.float32, self.goal_size),
            ("noisy_actions", np.float32, self.action_size),
            ("_simulator_index", np.int16)
        ])
        self._policy_data_buffer = np.zeros(
            shape=(self.episode_length, self.n_simulations),
            dtype=self._policy_data_type
        )
        self._policy_data_buffer["_simulator_index"] = np.arange(
            self.n_simulations)
        # critic
        self._critic_data_type = np.dtype([
            ("states", np.float32, self.state_size),
            ("goals", np.float32, self.goal_size),
            ("current_goals", np.float32, self.goal_size),
            ("pure_actions", np.float32, self.action_size),
            ("predictions", np.float32, self.prediction_size),
            ("targets", np.float32, self.prediction_size),
            ("integer_targets", np.int16),
            ("_simulator_index", np.int16)
        ])
        self._critic_data_buffer = np.zeros(
            shape=(self.episode_length, self.n_simulations),
            dtype=self._critic_data_type
        )
        self._critic_data_buffer["_simulator_index"] = np.arange(
            self.n_simulations)
        # COUNTERS
        self.n_policy_episodes = 0
        self.n_critic_episodes = 0
        self.n_policy_transition_gathered = 0
        self.n_critic_transition_gathered = 0
        self.n_policy_training = 0
        self.n_critic_training = 0

        # TENSORBOARD LOGGING
        self.tb_training_policy_loss = Mean(
            "training/policy_loss",
            dtype=tf.float32
        )
        self.tb_training_critic_loss = Mean(
            "training/critic_loss",
            dtype=tf.float32
        )
        self.tb_collection_policy_it_per_sec = Mean(
            "collection/policy_it_per_sec",
            dtype=tf.float32
        )
        self.tb_collection_critic_it_per_sec = Mean(
            "collection/critic_it_per_sec",
            dtype=tf.float32
        )
        self.tb_collection_critic_mean_prediction_error = Mean(
            "collection/critic_mean_prediction_error",
            dtype=tf.float32
        )
        self.tb_collection_critic_mean_prediction = Mean(
            "collection/critic_mean_prediction",
            dtype=tf.float32
        )
        self.tb_collection_critic_success_rate_percent = Mean(
            "collection/critic_success_rate_percent",
            dtype=tf.float32
        )
        self.tb_collection_policy_success_rate_percent = Mean(
            "collection/policy_success_rate_percent",
            dtype=tf.float32
        )
        self.tb_collection_policy_discoveries_per_ep = Mean(
            "collection/policy_discoveries_per_ep",
            dtype=tf.float32
        )
        self.summary_writer = tf.summary.create_file_writer("logs")
        with self.summary_writer.as_default():
            hp.hparams(self._hparams)

    def log_summaries(self, critic=True, policy=True):
        with self.summary_writer.as_default():
            training_critic_metric_list = [
                self.tb_training_critic_loss,
            ]
            collection_critic_metric_list = [
                self.tb_collection_critic_it_per_sec,
                self.tb_collection_critic_mean_prediction_error,
                self.tb_collection_critic_mean_prediction,
                self.tb_collection_critic_success_rate_percent,
            ]
            training_policy_metric_list = [
                self.tb_training_policy_loss,
            ]
            collection_policy_metric_list = [
                self.tb_collection_policy_it_per_sec,
                self.tb_collection_policy_success_rate_percent,
                self.tb_collection_policy_discoveries_per_ep,
            ]
            if critic:
                for metric in training_critic_metric_list:
                    tf.summary.scalar(
                        metric.name,
                        metric.result(),
                        step=self.n_critic_episodes
                    )
                    metric.reset_states()
                for metric in collection_critic_metric_list:
                    tf.summary.scalar(
                        metric.name,
                        metric.result(),
                        step=self.n_critic_episodes
                    )
                    metric.reset_states()
            if policy:
                for metric in training_policy_metric_list:
                    tf.summary.scalar(
                        metric.name,
                        metric.result(),
                        step=self.n_policy_episodes
                    )
                    metric.reset_states()
                for metric in collection_policy_metric_list:
                    tf.summary.scalar(
                        metric.name,
                        metric.result(),
                        step=self.n_policy_episodes
                    )
                    metric.reset_states()

    def _get_n_episodes(self):
        return self.n_policy_episodes + self.n_critic_episodes
    n_episodes = property(_get_n_episodes)

    def _get_current_policy_ratio(self):
        if self.n_policy_transition_gathered != 0:
            return self.n_policy_training * \
                    self.batch_size / \
                    self.n_policy_transition_gathered
        else:
            return np.inf
    current_policy_ratio = property(_get_current_policy_ratio)

    def _get_current_critic_ratio(self):
        if self.n_critic_transition_gathered != 0:
            return self.n_critic_training * \
                    self.batch_size / \
                    self.n_critic_transition_gathered
        else:
            return np.inf
    current_critic_ratio = property(_get_current_critic_ratio)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        self.simulation_pool.close()

    def save(self):
        """Saves the model in the appropriate directory"""
        path = "./checkpoints/{:08d}".format(self.n_episodes)
        self.agent.save_weights(path)

    def restore(self, path):
        """Restores the weights from a checkpoint"""
        self.agent.load_weights(path)

    def sample_goals(self):
        """Returns a binary vector corresponding to the goal states of the
        actuators in the simulation for each simulation"""
        return np.random.randint(2, size=(self.n_simulations, self.goal_size))

    def _convert_to_integers(self, predictions, mode="sum"):
        if mode == "sum":
            return predictions.shape[-1] - np.sum(predictions, axis=-1)
        else:
            raise ValueError("Unrecognized convertion mode ({})".format(mode))

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
                goals = self.sample_goals()
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

    def gather_policy_data(self):
        """Performs one episode of exploration, places data in the policy
        buffer"""
        goals = self.sample_goals()
        states, current_goals = self.reset_simulations()
        time_start = time.time()
        for iteration in range(self.episode_length):
            pure_actions, noisy_actions, noises = self.agent.get_actions(
                states, goals, exploration=True)
            self._policy_data_buffer[iteration]["states"] = states
            self._policy_data_buffer[iteration]["goals"] = goals
            self._policy_data_buffer[iteration]["current_goals"] = current_goals
            self._policy_data_buffer[iteration]["noisy_actions"] = noisy_actions
            states, current_goals = self.apply_action(noisy_actions)
        # SELECT TRANSITIONS TO BE ADDED TO THE REPLAY BUFFER
        integer_predictions = self._convert_to_integers(
            self._policy_data_buffer["predictions"])
        advantages = np.apply_along_axis(
            lambda x: np.convolve(x, self.prediction_filter, mode='valid'),
            axis=0,
            arr=integer_predictions
        ) - (self.prediction_filter_lookup + 1) / 2
        where = advantages > 0
        self.policy_buffer.integrate(
            self._policy_data_buffer[:-self.prediction_filter_lookup][where]
        )
        n_discoveries = np.sum(where)
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


    def get_data(self):
        states, current_goals = tuple(zip(*self.simulation_pool.get_data()))
        return np.vstack(states), np.vstack(current_goals)

    def apply_action(self, actions):
        with self.simulation_pool.distribute_args():
            states, current_goals = \
                tuple(zip(*self.simulation_pool.apply_action(actions)))
        return np.vstack(states), np.vstack(current_goals)

    def gather_critic_data(self):
        """Performs one episode of testing, places the data in the critic
        buffer"""
        time_start = time.time()
        goals = self.sample_goals()
        states, current_goals = self.reset_simulations()
        for iteration in range(self.episode_length):
            pure_actions = self.agent.get_actions(
                states, goals, exploration=False)
            predictions = self.agent.get_predictions(states, goals)
            self._critic_data_buffer[iteration]["states"] = states
            self._critic_data_buffer[iteration]["goals"] = goals
            self._critic_data_buffer[iteration]["current_goals"] = current_goals
            self._critic_data_buffer[iteration]["pure_actions"] = pure_actions
            self._critic_data_buffer[iteration]["predictions"] = predictions
            states, current_goals = self.apply_action(pure_actions)
        # COMPLETE THE BUFFER WITH THE PREDICTION TARGETS
        bootstraping_predictions = self.agent.get_predictions(states, goals)
        prev = self._convert_to_integers(bootstraping_predictions)
        goal_missed = (
            self._critic_data_buffer["goals"] != \
            self._critic_data_buffer["current_goals"]
        ).any(axis=-1)
        failed_episode = goal_missed.all(axis=0)
        prev[failed_episode] = self.prediction_size
        self._critic_data_buffer["integer_targets"] = 0
        for i in range(self.episode_length - 1, -1, -1):
            prev = (prev + 1) * goal_missed[i]
            self._critic_data_buffer[i]["integer_targets"] = prev
        # CONVERT INTEGER TO TARGET
        self._critic_data_buffer["targets"] = 0
        for i in range(self.episode_length):
            for j in range(self.n_simulations):
                integer_target = self._critic_data_buffer["integer_targets"][i, j]
                self._critic_data_buffer["targets"][i, j, integer_target:] = 1
        self.critic_buffer.integrate(self._critic_data_buffer)
        self.n_critic_transition_gathered += self.episode_length * self.n_simulations
        self.n_critic_episodes += self.n_simulations
        time_stop = time.time()
        self.tb_collection_critic_it_per_sec(
            self.episode_length * self.n_simulations / (time_stop - time_start)
        )
        integer_predictions = self._convert_to_integers(
            self._critic_data_buffer["predictions"]
        )
        self.tb_collection_critic_mean_prediction(
            np.mean(integer_predictions)
        )
        cliped_integer_targets = np.clip(
            self._critic_data_buffer["integer_targets"],
            0,
            self.prediction_size
        )
        self.tb_collection_critic_mean_prediction_error(
            np.mean(np.abs(
            integer_predictions - cliped_integer_targets
            ))
        )
        self.tb_collection_critic_success_rate_percent(
            100 * np.mean(1 - failed_episode)
        )
        index_visualize = np.argmin(failed_episode) # index sucess episode if any
        if self._visualization is not None:
            self._visualization(
                target=cliped_integer_targets[:, index_visualize],
                prediction=integer_predictions[:, index_visualize]
            )

    def train_policy(self):
        """Performs one iteration of policy training (one weight update)"""
        data = self.policy_buffer.sample(self.batch_size)
        loss = self.agent.train_policy(
            data["states"],
            data["goals"],
            data["noisy_actions"]
        )
        self.n_policy_training += 1
        self.tb_training_policy_loss(loss)
        return loss

    def train_critic(self):
        """Performs one iteration of critic training (one weight update)"""
        data = self.critic_buffer.sample(self.batch_size)
        loss = self.agent.train_critic(
            data["states"],
            data["goals"],
            data["targets"]
        )
        self.n_critic_training += 1
        self.tb_training_critic_loss(loss)
        return loss

    def gather_and_train_policy(self):
        """Gathers one episode o training data and train, according to the ratio
        policy_updates_per_sample"""
        self.gather_policy_data()
        while self.current_policy_ratio < self.policy_updates_per_sample:
            self.train_policy()

    def gather_and_train_critic(self):
        """Gathers one episode o training data and train, according to the ratio
        critic_updates_per_sample"""
        self.gather_critic_data()
        while self.current_critic_ratio < self.critic_updates_per_sample:
            self.train_critic()

    def gather_train_and_log(self, critic=True, policy=True):
        if critic:
            self.gather_and_train_critic()
        if policy:
            self.gather_and_train_policy()
        if self.n_critic_episodes % 1 == 0:
            self.log_summaries(critic=critic, policy=policy)

    def evaluate(self):
        """Evaluates the current model"""
        pass

    def run_script(self, script):
        """Runs the script (exploration, evaluation, training, ploting, saving
        etc) according to the script"""
        pass

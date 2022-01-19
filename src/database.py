import sqlite3 as sql
import pandas as pd
from datetime import datetime


class Database:
    def __init__(self, path):
        print('[database] opening {}'.format(path))
        self.path = path
        self.conn = sql.connect(path, detect_types=sql.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        command = '''CREATE TABLE IF NOT EXISTS experiments (
                     experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     date_time DATETIME NOT NULL,
                     collection TEXT NOT NULL,
                     policy_model_arch TEXT NOT NULL,
                     critic_model_arch TEXT NOT NULL,
                     policy_primitive_learning_rate FLOAT NOT NULL,
                     policy_movement_learning_rate FLOAT NOT NULL,
                     primitive_exploration_stddev FLOAT NOT NULL,
                     movement_exploration_stddev FLOAT NOT NULL,
                     critic_learning_rate FLOAT NOT NULL,
                     target_smoothing_stddev FLOAT NOT NULL,
                     tau FLOAT NOT NULL,
                     exploration_prob FLOAT NOT NULL,
                     state_size INTEGER NOT NULL,
                     action_size INTEGER NOT NULL,
                     goal_size INTEGER NOT NULL,
                     n_simulations INTEGER NOT NULL,
                     movement_exploration_prob_ratio FLOAT NOT NULL,
                     policy_bottleneck_size INTEGER NOT NULL,
                     policy_default_layer_size INTEGER NOT NULL,
                     critic_default_layer_size INTEGER NOT NULL,
                     environement TEXT NOT NULL,
                     has_movement_primitive INTEGER NOT NULL,
                     evaluation_goals TEXT NOT NULL,
                     exploration_goals TEXT NOT NULL,
                     episode_length_in_sec FLOAT NOT NULL,
                     episode_length_in_it INTEGER NOT NULL,
                     movement_mode TEXT NOT NULL,
                     discount_factor FLOAT NOT NULL,
                     movement_discount_factor FLOAT NOT NULL,
                     primitive_discount_factor FLOAT,
                     n_actions_in_movement INTEGER NOT NULL,
                     one_action_span_in_sec FLOAT NOT NULL,
                     simulation_timestep FLOAT NOT NULL,
                     updates_per_sample INTEGER NOT NULL,
                     batch_size INTEGER NOT NULL,
                     her_max_replays INTEGER NOT NULL,
                     movement_noise_magnitude_limit FLOAT NOT NULL,
                     primitive_noise_magnitude_limit FLOAT NOT NULL,
                     metabolic_cost_scale FLOAT NOT NULL,
                     buffer_size INTEGER NOT NULL,
                     path TEXT NOT NULL,
                     finished INTEGER NOT NULL
                  );'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS results (
                     experiment_id INTEGER NOT NULL,
                     episode_nb INTEGER NOT NULL,
                     training_step INTEGER NOT NULL,
                     it_per_sec FLOAT NOT NULL,
                     success_rate_percent FLOAT NOT NULL,
                     time_to_solve FLOAT,
                     diversity_per_ep FLOAT NOT NULL,
                     delta_distance_to_goal FLOAT NOT NULL,
                     n_register_change FLOAT NOT NULL,
                     one_away_sucess_rate FLOAT NOT NULL,
                     metabolic_cost FLOAT NOT NULL,
                     movement_critic_snr FLOAT NOT NULL,
                     primitive_critic_snr FLOAT
                  );'''
        self.cursor.execute(command)

    def insert_experiment(self,
            date_time,
            collection,
            policy_model_arch,
            critic_model_arch,
            policy_primitive_learning_rate,
            policy_movement_learning_rate,
            primitive_exploration_stddev,
            movement_exploration_stddev,
            critic_learning_rate,
            target_smoothing_stddev,
            tau,
            exploration_prob,
            state_size,
            action_size,
            goal_size,
            n_simulations,
            movement_exploration_prob_ratio,
            policy_bottleneck_size,
            policy_default_layer_size,
            critic_default_layer_size,
            environement,
            has_movement_primitive,
            evaluation_goals,
            exploration_goals,
            episode_length_in_sec,
            episode_length_in_it,
            movement_mode,
            discount_factor,
            movement_discount_factor,
            primitive_discount_factor,
            n_actions_in_movement,
            one_action_span_in_sec,
            simulation_timestep,
            updates_per_sample,
            batch_size,
            her_max_replays,
            movement_noise_magnitude_limit,
            primitive_noise_magnitude_limit,
            metabolic_cost_scale,
            buffer_size,
            path,
            finished):
        params = (
            date_time,
            collection,
            policy_model_arch,
            critic_model_arch,
            policy_primitive_learning_rate,
            policy_movement_learning_rate,
            primitive_exploration_stddev,
            movement_exploration_stddev,
            critic_learning_rate,
            target_smoothing_stddev,
            tau,
            exploration_prob,
            state_size,
            action_size,
            goal_size,
            n_simulations,
            movement_exploration_prob_ratio,
            policy_bottleneck_size,
            policy_default_layer_size,
            critic_default_layer_size,
            environement,
            has_movement_primitive,
            evaluation_goals,
            exploration_goals,
            episode_length_in_sec,
            episode_length_in_it,
            movement_mode,
            discount_factor,
            movement_discount_factor,
            primitive_discount_factor,
            n_actions_in_movement,
            one_action_span_in_sec,
            simulation_timestep,
            updates_per_sample,
            batch_size,
            her_max_replays,
            movement_noise_magnitude_limit,
            primitive_noise_magnitude_limit,
            metabolic_cost_scale,
            buffer_size,
            path,
            finished
        )
        command = '''INSERT INTO experiments(
                     date_time,
                     collection,
                     policy_model_arch,
                     critic_model_arch,
                     policy_primitive_learning_rate,
                     policy_movement_learning_rate,
                     primitive_exploration_stddev,
                     movement_exploration_stddev,
                     critic_learning_rate,
                     target_smoothing_stddev,
                     tau,
                     exploration_prob,
                     state_size,
                     action_size,
                     goal_size,
                     n_simulations,
                     movement_exploration_prob_ratio,
                     policy_bottleneck_size,
                     policy_default_layer_size,
                     critic_default_layer_size,
                     environement,
                     has_movement_primitive,
                     evaluation_goals,
                     exploration_goals,
                     episode_length_in_sec,
                     episode_length_in_it,
                     movement_mode,
                     discount_factor,
                     movement_discount_factor,
                     primitive_discount_factor,
                     n_actions_in_movement,
                     one_action_span_in_sec,
                     simulation_timestep,
                     updates_per_sample,
                     batch_size,
                     her_max_replays,
                     movement_noise_magnitude_limit,
                     primitive_noise_magnitude_limit,
                     metabolic_cost_scale,
                     buffer_size,
                     path,
                     finished)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        command_get_id = "SELECT last_insert_rowid()"
        with self.conn:
            self.cursor.execute(command, params)
            self.commit()
            self.cursor.execute(command_get_id)
            return self.cursor.fetchone()[0]

    def insert_result(self,
            experiment_id,
            episode_nb,
            training_step,
            it_per_sec,
            success_rate_percent,
            time_to_solve,
            diversity_per_ep,
            delta_distance_to_goal,
            n_register_change,
            one_away_sucess_rate,
            metabolic_cost,
            movement_critic_snr,
            primitive_critic_snr=None):
        params = (
            experiment_id,
            episode_nb,
            training_step,
            it_per_sec,
            success_rate_percent,
            time_to_solve,
            diversity_per_ep,
            delta_distance_to_goal,
            n_register_change,
            one_away_sucess_rate,
            metabolic_cost,
            movement_critic_snr,
            primitive_critic_snr
        )
        command = '''INSERT INTO results(
            experiment_id,
            episode_nb,
            training_step,
            it_per_sec,
            success_rate_percent,
            time_to_solve,
            diversity_per_ep,
            delta_distance_to_goal,
            n_register_change,
            one_away_sucess_rate,
            metabolic_cost,
            movement_critic_snr,
            primitive_critic_snr)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        self.cursor.execute(command, params)
        self.commit()

    def register_termination(self, experiment_id):
        command = '''UPDATE experiments SET finished=1 WHERE experiment_id=(?)'''
        self.cursor.execute(command, (experiment_id,))
        self.commit()

    def delete_exp(self, experiment_id):
        print(f'[database] deleting experiment with id {experiment_id}')
        command = 'DELETE FROM experiments WHERE experiment_id=?'
        self.cursor.execute(command, (experiment_id,))
        command = 'DELETE FROM results WHERE experiment_id=?'
        self.cursor.execute(command, (experiment_id,))
        self.commit()

    def delete_collection(self, collection):
        print(f'[database] deleting collection {collection}')
        ids = self.get_experiment_ids(collection=collection)
        for ex_id in ids:
            self.delete_exp(ex_id)

    def get_dataframe(self, command):
        return pd.read_sql(command, self.conn)

    def get_experiment_ids(self, **kwargs):
        where = tuple(f"{key} IS NULL" if value is None else f"{key}='{value}'" for key, value in kwargs.items())
        where = ' AND\n            '.join(where)
        command = '''
        SELECT
            experiment_id,
            finished
        FROM
            experiments''' + (f'''
        WHERE
            {where}
        ''' if kwargs else '')
        self.cursor.execute(command)
        results = self.cursor.fetchall()
        ret = tuple(map(lambda x: x[0], results))
        # print(kwargs, len(ret), tuple(map(lambda x: x[1], results)))
        return ret

    def get_matching_experiments_ids(self, other_database, experiment_id):
        command = f'''
        SELECT
            policy_model_arch,
            critic_model_arch,
            policy_primitive_learning_rate,
            policy_movement_learning_rate,
            primitive_exploration_stddev,
            movement_exploration_stddev,
            critic_learning_rate,
            target_smoothing_stddev,
            tau,
            exploration_prob,
            state_size,
            action_size,
            goal_size,
            movement_exploration_prob_ratio,
            policy_bottleneck_size,
            policy_default_layer_size,
            critic_default_layer_size,
            environement,
            has_movement_primitive,
            evaluation_goals,
            exploration_goals,
            episode_length_in_sec,
            /*episode_length_in_it,*/
            movement_mode,
            discount_factor,
            movement_discount_factor,
            primitive_discount_factor,
            n_actions_in_movement,
            one_action_span_in_sec,
            simulation_timestep,
            updates_per_sample,
            batch_size,
            her_max_replays,
            movement_noise_magnitude_limit,
            primitive_noise_magnitude_limit,
            metabolic_cost_scale,
            buffer_size
        FROM
            experiments
        WHERE
            experiment_id='{experiment_id}'
        '''
        df = self.get_dataframe(command) #.fillna("missing")
        where = df.to_dict(orient='records')[0]
        experiment_ids = other_database.get_experiment_ids(**where)
        return experiment_ids

    def group_by_matching_experiments(self):
        experiment_ids = set(self.get_experiment_ids())
        groups = []
        while experiment_ids:
            experiment_id = experiment_ids.pop()
            matchings = self.get_matching_experiments_ids(self, experiment_id)
            groups.append(matchings)
            for matching in matchings:
                experiment_ids.discard(matching)
        return groups

    def get_collection_variation_parameters(self, collection):
        command = '''
        SELECT
            policy_model_arch,
            critic_model_arch,
            policy_primitive_learning_rate,
            policy_movement_learning_rate,
            primitive_exploration_stddev,
            movement_exploration_stddev,
            critic_learning_rate,
            target_smoothing_stddev,
            tau,
            exploration_prob,
            state_size,
            action_size,
            goal_size,
            movement_exploration_prob_ratio,
            policy_bottleneck_size,
            policy_default_layer_size,
            critic_default_layer_size,
            environement,
            has_movement_primitive,
            evaluation_goals,
            exploration_goals,
            episode_length_in_sec,
            /*episode_length_in_it,*/
            movement_mode,
            discount_factor,
            movement_discount_factor,
            primitive_discount_factor,
            n_actions_in_movement,
            one_action_span_in_sec,
            simulation_timestep,
            updates_per_sample,
            batch_size,
            her_max_replays,
            movement_noise_magnitude_limit,
            primitive_noise_magnitude_limit,
            metabolic_cost_scale,
            buffer_size
        FROM
            experiments
        WHERE
            collection='{}'
        '''.format(collection)
        df = self.get_dataframe(command).fillna("missing")
        df = df.loc[:, (df != df.iloc[0]).any()]
        if len(df.columns) > 0:
            frame = df.value_counts(sort=False).index.to_frame(index=False)
            return frame.sort_values(by=list(frame.columns))
        else:
            return pd.DataFrame({
                "date_time":
                [
                    self.get_experiment_field(experiment_id, "date_time")
                    for experiment_id in self.get_experiment_ids(collection=collection)
                ]
            })
        # # self.cursor.execute(command, (collection,))
        # # results = self.cursor.fetchall()
        #
        # results_unique = sorted(list(set(results)))
        # print('results_unique', results_unique)
        # transposed = list(zip(*results_unique))

    def get_experiment_field(self, experiment_id, field):
        df = self.get_dataframe(f"SELECT {field} FROM experiments WHERE experiment_id='{experiment_id}'")
        return df[field].values[0]

    def import_from_other_db(self, database, experiment_id, collection):
        command = 'SELECT * FROM experiments WHERE experiment_id=?'
        database.cursor.execute(command, (experiment_id,))
        res = database.cursor.fetchone()
        # print(f'found {res} in the other db')
        new_experiment_id = self.insert_experiment(datetime.now(), collection, *res[3:])
        # print(f'it will have the id {new_experiment_id} in this db')
        command = '''
        SELECT
            ?,
            episode_nb,
            training_step,
            it_per_sec,
            success_rate_percent,
            time_to_solve,
            diversity_per_ep,
            delta_distance_to_goal,
            n_register_change,
            one_away_sucess_rate,
            metabolic_cost,
            movement_critic_snr,
            primitive_critic_snr
        FROM
            results
        WHERE
            experiment_id=?
        '''
        database.cursor.execute(command, (new_experiment_id, experiment_id))
        for row in database.cursor:
            # print(f'inserting row {row}')
            self.cursor.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', row)
        self.commit()
        print(f"[database] imported experiment {experiment_id} from {database.path} to {self.path} in collection {collection} -- new experiment_id {new_experiment_id}")
        return new_experiment_id

    def get_experiment_last_episode_nb(self, experiment_id):
        command = f'''
        SELECT
            MAX(episode_nb)
        FROM
            results
        WHERE
            experiment_id='{experiment_id}'
        '''
        self.cursor.execute(command)
        res = self.cursor.fetchone()
        return res[0]

    def copy_to_new_collection(self, experiment_id, collection):
        command = 'SELECT * FROM experiments WHERE experiment_id=?'
        self.cursor.execute(command, (experiment_id,))
        res = self.cursor.fetchone()
        new_experiment_id = self.insert_experiment(datetime.now(), collection, *res[3:])
        command = '''
        INSERT INTO results(
            experiment_id,
            episode_nb,
            training_step,
            it_per_sec,
            success_rate_percent,
            time_to_solve,
            diversity_per_ep,
            delta_distance_to_goal,
            n_register_change,
            one_away_sucess_rate,
            metabolic_cost,
            movement_critic_snr,
            primitive_critic_snr)
        SELECT
            ?,
            episode_nb,
            training_step,
            it_per_sec,
            success_rate_percent,
            time_to_solve,
            diversity_per_ep,
            delta_distance_to_goal,
            n_register_change,
            one_away_sucess_rate,
            metabolic_cost,
            movement_critic_snr,
            primitive_critic_snr
        FROM
            results
        WHERE
            experiment_id=?
        '''
        self.cursor.execute(command, (new_experiment_id, experiment_id))
        self.commit()

    def commit(self):
        self.conn.commit()



if __name__ == '__main__':
    db = Database('../databases/debug2.db')
    experiment_id = db.insert_experiment(
        date_time=datetime.now(),
        collection='text',
        policy_model_arch='string',
        critic_model_arch='string',
        policy_primitive_learning_rate=1.23,
        policy_movement_learning_rate=1.23,
        primitive_exploration_stddev=1.23,
        movement_exploration_stddev=1.23,
        critic_learning_rate=1.23,
        target_smoothing_stddev=1.23,
        tau=1.23,
        exploration_prob=1.23,
        state_size=456,
        action_size=456,
        goal_size=456,
        n_simulations=456,
        movement_exploration_prob_ratio=1.23,
        policy_bottleneck_size=456,
        policy_default_layer_size=456,
        critic_default_layer_size=456,
        environement='string',
        has_movement_primitive=456,
        evaluation_goals='string',
        exploration_goals='string',
        episode_length_in_sec=1.23,
        episode_length_in_it=456,
        movement_mode='string',
        discount_factor=1.23,
        movement_discount_factor=1.23,
        primitive_discount_factor=1.23,
        n_actions_in_movement=456,
        one_action_span_in_sec=1.23,
        simulation_timestep=1.23,
        updates_per_sample=456,
        batch_size=456,
        her_max_replays=456,
        movement_noise_magnitude_limit=1.23,
        primitive_noise_magnitude_limit=1.23,
        metabolic_cost_scale=1.23,
        buffer_size=456,
        path='string',
        finished=456,
    )

    print("##############")
    print(experiment_id)
    print("##############")

    db.insert_result(
         experiment_id=experiment_id,
         episode_nb=789,
         training_step=789,
         it_per_sec=10.11,
         success_rate_percent=10.11,
         time_to_solve=10.11,
         diversity_per_ep=10.11,
         delta_distance_to_goal=10.11,
         n_register_change=10.11,
         one_away_sucess_rate=10.11,
         metabolic_cost=10.11,
         movement_critic_snr=10.11,
         primitive_critic_snr=10.11
    )

    experiments = db.get_dataframe('SELECT * FROM experiments')
    print(experiments)
    print('\n')
    results = db.get_dataframe('SELECT * FROM results')
    print(results)
    print('\n')

    db.copy_to_new_collection(1, 'test_copy')

    db.register_termination(experiment_id)
    experiments = db.get_dataframe('SELECT * FROM experiments')
    print(experiments)
    print('\n')
    results = db.get_dataframe('SELECT * FROM results')
    print(results)
    print('\n')

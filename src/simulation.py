from pyrep import PyRep
import multiprocessing as mp
import os
from pathlib import Path
from collections import defaultdict
from custom_shapes import TapShape, ButtonShape, LeverShape, Kuka
import atexit
import numpy as np
from contextlib import contextmanager


def communicate_return_value(method):
    """methode from the SimulationConsumer class decorated with this function
    will send there return value to the SimulationProducer class"""
    method._communicate_return_value = True
    return method


def default_dont_communicate_return(cls):
    for attribute_name, attribute in cls.__dict__.items():
        if callable(attribute):
            communicate = hasattr(attribute, '_communicate_return_value')
            attribute._communicate_return_value = communicate
    return cls


def c2p_convertion_function(cls, method):
    def new_method(self, *args, **kwargs):
        cls._send_command(self, method, *args, **kwargs)
        if method._communicate_return_value:
            return self._process_io["return_value_pipe_out"].recv()
    return new_method


def consumer_to_producer_method_conversion(cls):
    proc_methods = [
        "run", "is_alive", "join", "kill", "start", "terminate", "close"
    ]
    method_dict = {
        **SimulationConsumerAbstract.__dict__,
        **SimulationConsumer.__dict__,
    }
    convertables = {
        method_name: method \
        for method_name, method in method_dict.items()\
        if callable(method) and\
        method_name not in proc_methods and\
        not method_name.startswith("_")
    }
    for method_name, method in convertables.items():
        new_method = c2p_convertion_function(cls, method)
        setattr(cls, method_name, new_method)
    return cls


def p2p_convertion_function(name):
    def new_method(self, *args, **kwargs):
        if self._distribute_args_mode:
            # all args are iterables that must be distributed to each producer
            return [
                getattr(producer, name)(
                    *[arg[i] for arg in args],
                    **{key: value[i] for key, value in kwargs.items()}
                )
                for producer in enumerate(self._active_producers)
            ]
        else:
            return [
                getattr(producer, name)(*args, **kwargs)
                for producer in self._active_producers
            ]
    return new_method

def producer_to_pool_method_convertion(cls):
    convertables = {
        method_name: method \
        for method_name, method in SimulationProducer.__dict__.items()\
        if callable(method) and not method_name.startswith("_")
    }
    for method_name, method in convertables.items():
        new_method = p2p_convertion_function(method_name)
        setattr(cls, method_name, new_method)
    return cls


@default_dont_communicate_return
class SimulationConsumerAbstract(mp.Process):
    _id = 0
    """This class sole purpose is to better 'hide' all interprocess related code
    from the user."""
    def __init__(self, process_io, scene="", gui=False):
        super().__init__(
            name="simulation_consumer_{}".format(SimulationConsumerAbstract._id)
        )
        self._id = SimulationConsumerAbstract._id
        SimulationConsumerAbstract._id += 1
        self._scene = scene
        self._gui = gui
        self._process_io = process_io

    def run(self):
        self._pyrep = PyRep()
        self._pyrep.launch(
            self._scene,
            headless=not self._gui,
            write_coppeliasim_stdout_to_file=True
        )
        self._process_io["simulaton_ready"].set()
        self._main_loop()

    def _main_loop(self):
        while not self._process_io["must_quit"].is_set():
            self._consume_command()
        self._pyrep.shutdown()

    def _consume_command(self):
        command = self._process_io["command_pipe_out"].recv()
        self._process_io["slot_in_command_queue"].release()
        ret = command[0](self, *command[1], **command[2])
        if command[0]._communicate_return_value:
            self._communicate_return_value(ret)

    def _communicate_return_value(self, value):
        self._process_io["return_value_pipe_in"].send(value)

    def signal_command_pipe_empty(self):
        self._process_io["command_pipe_empty"].set()
        self._process_io["command_pipe_empty"].clear()

    def good_bye(self):
        pass


@default_dont_communicate_return
class SimulationConsumer(SimulationConsumerAbstract):
    def __init__(self, process_io, scene="", gui=False):
        super().__init__(process_io, scene, gui)
        self._shapes = defaultdict(list)
        self._stateful_shape_list = []
        self._arm_list = []

    @communicate_return_value
    def get_stateful_objects_states(self):
        for i, shape in enumerate(self._stateful_shape_list):
            self._stateful_shape_state_buffer[i] = shape.get_state()
        # todo: return value must be communicated out of the process
        return self._stateful_shape_state_buffer

    def set_stateful_objects_states(self, states):
        if len(states) != len(self._stateful_shape_list):
            raise ValueError("Can not set the object states, wrong length")
        for shape, state in zip(self._stateful_shape_list, states):
            shape.set_state(state)

    def _add_stateful_object(self, model):
        self._stateful_shape_list.append(model)
        self._stateful_shape_state_buffer = np.zeros(
            len(self._stateful_shape_list),
            dtype=np.uint8
        )

    def add_tap(self, position=None, orientation=None):
        model = self._pyrep.import_model("../3d_models/tap.ttm")
        model = TapShape(model.get_handle(), self._pyrep)
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["tap"].append(model)
        self._add_stateful_object(model)

    def add_button(self, position=None, orientation=None):
        model = self._pyrep.import_model("../3d_models/button.ttm")
        model = ButtonShape(model.get_handle(), self._pyrep)
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["button"].append(model)
        self._add_stateful_object(model)

    def add_lever(self, position=None, orientation=None):
        model = self._pyrep.import_model("../3d_models/lever.ttm")
        model = LeverShape(model.get_handle(), self._pyrep)
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["lever"].append(model)
        self._add_stateful_object(model)

    def add_arm(self, position=None, orientation=None, from_tech_sheet=False):
        if from_tech_sheet:
            model_file = "../3d_models/kuka_from_tech_sheet.ttm"
        else:
            model_file = "../3d_models/kuka_default.ttm"
        model = self._pyrep.import_model(model_file)
        model = Kuka(model.get_handle())
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["arm"].append(model)
        self._arm_list.append(model)
        self._arm_joints_count = [arm.get_joint_count() for arm in self._arm_list]
        self._n_joints = sum(self._arm_joints_count)
        self._arm_joints_positions_buffer = np.zeros(
            self._n_joints,
            dtype=np.float32
        )
        self._arm_joints_velocities_buffer = np.zeros(
            self._n_joints,
            dtype=np.float32
        )
        self._arm_joints_torques_buffer = np.zeros(
            self._n_joints,
            dtype=np.float32
        )

    def _get_joint_positions(self):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            self._arm_joints_positions_buffer[last:next] = \
                arm.get_joint_positions()
            last = next
        return self._arm_joints_positions_buffer

    @communicate_return_value
    def get_joint_positions(self):
        return self._get_joint_positions()

    def _get_joint_velocities(self):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            self._arm_joints_velocities_buffer[last:next] = \
                arm.get_joint_velocities()
            last = next
        return self._arm_joints_velocities_buffer

    @communicate_return_value
    def get_joint_velocities(self):
        return self._get_joint_velocities()

    def set_joint_target_velocities(self, velocities):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            arm.set_joint_target_velocities(velocities[last:next])
            last = next

    def set_control_loop_enabled(self, bool):
        for arm in self._arm_list:
            arm.set_control_loop_enabled(bool)

    def set_motor_locked_at_zero_velocity(self, bool):
        for arm in self._arm_list:
            arm.set_motor_locked_at_zero_velocity(bool)

    def _get_joint_forces(self):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            self._arm_joints_torques_buffer[last:next] = \
                arm.get_joint_forces()
            last = next
        return self._arm_joints_torques_buffer

    @communicate_return_value
    def get_joint_forces(self):
        return self._get_joint_forces()

    def _get_joint_upper_velocity_limits(self):
        last = 0
        next = 0
        upper_velocity_limits = np.zeros(self._n_joints, dtype=np.float32)
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            upper_velocity_limits[last:next] = \
                arm.get_joint_upper_velocity_limits()
            last = next
        return upper_velocity_limits

    @communicate_return_value
    def get_joint_upper_velocity_limits(self):
        return self._get_joint_upper_velocity_limits()

    @communicate_return_value
    def get_n_joints(self):
        return self._n_joints

    def create_environment(self, type='one_arm_4_buttons'):
        if type == 'one_arm_4_buttons':
            self.add_arm()
            distance = 0.65
            self.add_button(position=( distance, 0, 0))
            self.add_button(position=(-distance, 0, 0))
            self.add_button(position=(0,  distance, 0))
            self.add_button(position=(0, -distance, 0))
        elif type == 'one_arm_2_buttons_2_levers':
            self.add_arm()
            distance = 0.65
            self.add_button(position=( distance, 0, 0))
            self.add_button(position=(-distance, 0, 0))
            self.add_lever(position=(0,  distance, 0))
            self.add_lever(position=(0, -distance, 0))
        elif type == 'one_arm_2_buttons_1_levers_1_tap':
            self.add_arm()
            distance = 0.65
            self.add_button(position=( distance, 0, 0))
            self.add_button(position=(-distance, 0, 0))
            self.add_lever(position=(0,  distance, 0))
            self.add_tap(position=(0, -distance, 0))
        else:
            raise ValueError("Unrecognized environment type ({})".format(type))

    def step_sim(self):
        self._pyrep.step()

    def start_sim(self):
        self._pyrep.start()

    def stop_sim(self):
        self._pyrep.stop()

    def get_simulation_timestep(self):
        return self._pyrep.get_simulation_timestep()



@consumer_to_producer_method_conversion
class SimulationProducer(object):
    def __init__(self, scene="", gui=False):
        self._process_io = {}
        self._process_io["must_quit"] = mp.Event()
        self._process_io["simulaton_ready"] = mp.Event()
        self._process_io["command_pipe_empty"] = mp.Event()
        self._process_io["slot_in_command_queue"] = mp.Semaphore(100)
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["command_pipe_in"] = pipe_in
        self._process_io["command_pipe_out"] = pipe_out
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["return_value_pipe_in"] = pipe_in
        self._process_io["return_value_pipe_out"] = pipe_out
        self._consumer = SimulationConsumer(self._process_io, scene, gui=gui)
        self._consumer.start()
        print("consumer {} started".format(self._consumer._id))
        self._process_io["simulaton_ready"].wait()
        self._closed = False
        atexit.register(self.close)

    def _get_process_io(self):
        return self._process_io

    def _send_command(self, function, *args, **kwargs):
        self._process_io["command_pipe_in"].send((function, args, kwargs))
        self._process_io["slot_in_command_queue"].acquire()

    def close(self):
        if not self._closed:
            # print("Producer closing")
            self.wait_command_pipe_empty()
            # print("command pipe empty, setting must_quit flag")
            self._process_io["must_quit"].set()
            # print("flushing command pipe")
            self.good_bye()
            self._closed = True
            # print("succesfully closed")
            self._consumer.join()
            print("consumer {} closed".format(self._consumer._id))
        else:
            print("{} already closed, doing nothing".format(self._consumer._id))

    def wait_command_pipe_empty(self):
        self._send_command(SimulationConsumer.signal_command_pipe_empty)
        self._process_io["command_pipe_empty"].wait()


@producer_to_pool_method_convertion
class SimulationPool:
    def __init__(self, size, scene="", guis=[]):
        self._producers = [
            SimulationProducer(scene, gui=i in guis) for i in range(size)
        ]
        self._active_producers_indices = list(range(size))
        self._distribute_args_mode = False

    @contextmanager
    def specific(self, list_or_int):
        _active_producers_indices_before = self._active_producers_indices
        indices = list_or_int if type(list_or_int) is list else [list_or_int]
        self._active_producers_indices = indices
        yield
        self._active_producers_indices = _active_producers_indices_before

    @contextmanager
    def distribute_args(self):
        self._distribute_args_mode = True
        yield
        self._distribute_args_mode = False

    def _get_active_producers(self):
        return [self._producers[i] for i in self._active_producers_indices]
    _active_producers = property(_get_active_producers)


if __name__ == '__main__':
    def test_1():
        scene = ""
        simulation = SimulationProducer(scene, gui=True)
        simulation.add_tap(position=(1, 1, 0), orientation=(0, 0, 1))
        simulation.add_tap(position=(2, 1, 0), orientation=(0, 0, 1))
        simulation.add_button(position=(0, 1, 0), orientation=(0, 0, 0))
        simulation.add_button(position=(0, 0, 0), orientation=(0, 0, 0))
        simulation.add_lever(position=(1, 0, 0), orientation=(0, 0, 0))
        simulation.add_lever(position=(2, 0, 0), orientation=(0, 0, 0))
        simulation.start_sim()

        for j in range(1):
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([1, 0, 0, 0, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 1, 0, 0, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 1, 0, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 0, 1, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 0, 0, 1, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 0, 0, 0, 1])

        print(simulation.get_stateful_objects_states())

        for i in range(5000):
            simulation.step_sim()
            print(i, end='\r')
        simulation.stop_sim()
        simulation.close()

    def test_2():
        simulations = SimulationPool(32)
        simulations.add_tap(position=(1, 1, 0), orientation=(0, 0, 1))
        simulations.add_tap(position=(2, 1, 0), orientation=(0, 0, 1))
        simulations.add_button(position=(0, 1, 0), orientation=(0, 0, 0))
        simulations.add_button(position=(0, 0, 0), orientation=(0, 0, 0))
        simulations.add_lever(position=(1, 0, 0), orientation=(0, 0, 0))
        simulations.add_lever(position=(2, 0, 0), orientation=(0, 0, 0))
        simulations.start_sim()
        simulations.set_stateful_objects_states([0, 0, 0, 0, 1, 0])
        print(simulations.get_stateful_objects_states())
        with simulations.specific(0):
            simulations.set_stateful_objects_states([0, 0, 0, 0, 1, 1])
        print(simulations.get_stateful_objects_states())
        simulations.stop_sim()
        return simulations

    def test_3():
        import time
        M = 32
        simulations = SimulationPool(M, guis=[])
        simulations.create_environment('one_arm_2_buttons_1_levers_1_tap')
        simulations.start_sim()
        simulations.step_sim()
        print(simulations.get_joint_positions())
        print(simulations.get_joint_velocities())
        print(simulations.get_joint_forces())
        print(simulations.get_joint_upper_velocity_limits())
        N = 1000
        t0 = time.time()
        for i in range(N):
            simulations.step_sim()
        t1 = time.time()
        print("Pool size: {}, {} iteration in {:.3f} sec ({:.3f} it/sec)".format(
            M,
            N * M,
            t1 - t0,
            M * N / (t1 - t0)
        ))
        simulations.stop_sim()
        simulations.close()

    def test_4():
        import time
        pool_size = 1
        simulations = SimulationPool(pool_size, guis=[0])
        # with simulations.same_argument_for_all():
        simulations.create_environment('one_arm_2_buttons_1_levers_1_tap')
        simulations.set_control_loop_enabled(False)
        simulations.start_sim()
        with simulations.specific(0):
            upper_limits = simulations.get_joint_upper_velocity_limits()[0]
            n_joints = simulations.get_n_joints()[0]

        N = 1000

        frequencies = np.random.randint(low=100, high=150, size=n_joints)[np.newaxis]
        x = np.arange(N)[:, np.newaxis]
        velocities = np.sin(x / frequencies * 2 * np.pi) * upper_limits

        t0 = time.time()

        for i in range(N):
            simulations.step_sim()
            simulations.set_joint_target_velocities(velocities[i])
            a = simulations.get_joint_forces()

        t1 = time.time()

        print("{} iteration in {:.3f} sec ({:.3f} it/sec)".format(
            N * pool_size,
            t1 - t0,
            N * pool_size / (t1 - t0)
        ))

        simulations.stop_sim()
        simulations.close()


    test_4()

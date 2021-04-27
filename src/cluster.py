import omegaconf
import hydra
import os
from hydra.utils import get_original_cwd
from paramiko import SSHClient, RSAKey, AutoAddPolicy
from getpass import getpass
from socket import gethostname
import sys
from collections import defaultdict
import json
import time
import custom_interpolations


PASSWORD = None
REMOTE_HOST_NAME = 'otto'


@hydra.main(config_path='../config/general', config_name='default.yaml')
def start_job(cfg):
        experiment_path = os.getcwd()
        pickle_conf_path = experiment_path + '/cfg.json'
        with open(pickle_conf_path, "w") as f:
            json.dump(omegaconf.OmegaConf.to_container(cfg, resolve=True), f, indent=4)
        command_line_args  = " rundir=" + experiment_path
        job_name = get_job_name()
        output_flag = "--output {outdir}/%N_%j.joblog".format(outdir=experiment_path)
        job_name_flag = "--job-name {job_name}".format(job_name=job_name)
        partition, reservation = get_partition_reservation()
        partition_flag = "--partition {partition}".format(partition=partition)
        reservation_flag = "--reservation {reservation}".format(reservation=reservation) if reservation is not None else ""
        os.chdir(get_original_cwd())
        command_line = "sbatch {output_flag} {job_name_flag} {partition_flag} {reservation_flag} cluster.sh ".format(
            output_flag=output_flag,
            job_name_flag=job_name_flag,
            partition_flag=partition_flag,
            reservation_flag=reservation_flag,
        ) + command_line_args
        print(command_line, flush=True)
        os.system(command_line)
        time.sleep(20)


def get_n_free_cpus(node):
    cpusstate = os.popen('sinfo -h --nodes {} -O cpusstate'.format(node)).read()
    cpusstate = cpusstate.split("/")
    return int(cpusstate[1])


def get_free_mem(node):
    meminfo = os.popen('sinfo -h --nodes {} -O memory,allocmem'.format(node)).read()
    memory, allocated_memory = [int(s) for s in meminfo.split() if s.isdigit()]
    return memory - allocated_memory


def get_n_free_gpus(node):
    total = os.popen("sinfo -h -p sleuths -n {} -O gres".format(node)).read()
    total = int(total.split(":")[-1])
    used = os.popen("squeue -h -w {} -O gres".format(node)).read()
    used = len(used.strip("\n").split("\n"))
    return total - used


def get_n_pending_job_per_option():
    info = os.popen("squeue -h -u wilmot -o '%P;%v;%t'").read().split('\n')
    ret = {'x-men;(null);PD': 0, 'sleuths;(null);PD': 0, 'sleuths;triesch-shared;PD': 0}
    for i in info:
        if i.split(';')[-1] == 'PD':
            ret[i] += 1
    return ret


def node_list_availability(node_list, min_cpus=8, min_free_mem=20000):
    for node in node_list:
        n_free_cpus = get_n_free_cpus(node)
        free_mem = get_free_mem(node)
        if n_free_cpus >= min_cpus and free_mem >= min_free_mem:
            print(node, end=" -> ")
            return True
    return False


def get_partition_reservation():
    n_pending_job_per_option = get_n_pending_job_per_option()
    # OPTION 2
    print("checking OPTION 2 ... ", end="")
    if node_list_availability(["turbine", "vane", "speedboat"]):
        print("free space available, sending job")
        if n_pending_job_per_option["sleuths;(null);PD"] == 0:
            return "sleuths", None
        else:
            print("there are pending jobs for that option")
    print("no free space")
    # OPTION 3
    print("checking OPTION 3 ... ", end="")
    if node_list_availability(["jetski"]):
        print("free space available, sending job")
        if n_pending_job_per_option["sleuths;triesch-shared;PD"] == 0:
            return "sleuths", "triesch-shared"
        else:
            print("there are pending jobs for that option")
    print("no free space")
    # OPTION 1
    print("checking OPTION 1 ... ", end="")
    if node_list_availability(["iceman", "jubilee", "frost", "beast", "cyclops", "shadowcat"]):
        print("free space available, sending job")
        if n_pending_job_per_option["x-men;(null);PD"] == 0:
            return "x-men", None
        else:
            print("there are pending jobs for that option")
    print("no free space")
    pending_job_target_ratio = {'x-men;(null);PD': 0.15, 'sleuths;(null);PD': 0.7, 'sleuths;triesch-shared;PD': 0.15}
    total = sum(n_pending_job_per_option.values())
    if total == 0:
        print("no pending job, sending to x-men")
        return "x-men", None
    pending_job_ratio = {key: value / total for key, value in n_pending_job_per_option.items()}
    print("current ratio of pending job is:", pending_job_ratio)
    print("target  ratio of pending job is:", pending_job_target_ratio)
    for key in pending_job_target_ratio:
        if pending_job_ratio[key] <= pending_job_target_ratio[key]:
            partition, reservation, _ = key.split(';')
            reservation = None if reservation == '(null)' else reservation
            print("sending on {} with reservation {}".format(partition, reservation))
            return partition, reservation
    print("Defaulting to x-men OPTION 1")
    return "x-men", None


def get_job_name():
    return os.path.basename(os.getcwd())


def ssh_command(cmd):
    global PASSWORD
    host="fias.uni-frankfurt.de"
    user="wilmot"
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.load_system_host_keys()
    # if PASSWORD is None:
    #     PASSWORD = getpass("Please enter password for the rsa key .ssh/id_rsa\n")
    # pkey = RSAKey.from_private_key_file("/home/cwilmot/.ssh/id_rsa", password=PASSWORD)
    # client.connect(host, username=user, pkey=pkey)
    PASSWORD = getpass("Please enter password\n")
    client.connect(host, username=user, password=PASSWORD)
    stdin, stdout, stderr = client.exec_command("""(
        eval "$(/home/wilmot/.software/miniconda/miniconda3/bin/conda shell.bash hook)" ;
        export COPPELIASIM_ROOT=/home/aecgroup/aecdata/Software/CoppeliaSim_4.0.0_rev4 ;
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT ;
        export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT ;
        export COPPELIASIM_MODEL_PATH=/home/wilmot/Documents/code/coppelia_sim_inverse_model/3d_models/ ;
        cd Documents/code/coppelia_sim_inverse_model/src ;
        {})""".format(cmd))
    for line in stdout.readlines():
        print(line, end="")
    for line in stderr.readlines():
        print(line, end="")
    print("")


if __name__ == "__main__" and gethostname() == REMOTE_HOST_NAME:
    start_job()
if __name__ == "__main__" and gethostname() != REMOTE_HOST_NAME:
    ssh_command("python " + " ".join(sys.argv))

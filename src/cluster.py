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


@hydra.main(config_path="../config/general", config_name='default.yaml')
def start_job(cfg):
        experiment_path = os.getcwd()
        conf_path = experiment_path + '/cfg.yaml'
        with open(conf_path, "w") as f:
            f.write(cfg.pretty(resolve=True))
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


def node_list_availability(node_list, min_cpus=10, min_free_mem=20000):
    for node in node_list:
        n_free_cpus = get_n_free_cpus(node)
        free_mem = get_free_mem(node)
        if n_free_cpus >= min_cpus and free_mem >= min_free_mem:
            print(node, end=" -> ")
            return True
    return False


def get_partition_reservation():
    # OPTION 1
    print("checking OPTION 1 ... ", end="")
    if node_list_availability(["xavier", "iceman", "jubilee", "frost", "beast", "cyclops", "shadowcat"]):
        print("free space available, sending job")
        return "x-men", None
    print("no free space")
    # OPTION 2
    print("checking OPTION 2 ... ", end="")
    if node_list_availability(["turbine", "vane"]):
        print("free space available, sending job")
        return "sleuths", None
    print("no free space")
    # OPTION 3
    print("checking OPTION 3 ... ", end="")
    if node_list_availability(["jetski"]):
        print("free space available, sending job")
        return "sleuths", "triesch-shared"
    print("no free space")
    print("No space available on the cluster. Defaulting to x-men OPTION 1")
    return "x-men", None


def node_to_n_jobs():
    nodes = os.popen(
        'squeue -h -u wilmot -O nodelist'
    ).read().strip('\n').replace(' ', '').split("\n")
    ret = defaultdict(int)
    for node in nodes:
        ret[node] += 1
    return ret


def reservation_to_n_jobs():
    reservations = os.popen(
        'squeue -h -u wilmot -O reservation'
    ).read().strip("\n").replace(' ', '').split("\n")
    ret = defaultdict(int)
    for reservation in reservations:
        ret[reservation] += 1
    return ret


def next_reservation():
    nodes_used = node_to_n_jobs()
    reservations = reservation_to_n_jobs()
    nodes = ["jetski", "turbine", "vane"]
    free_cpus = {node: get_n_free_cpus(node) for node in nodes}
    free_gpus = {node: get_n_free_gpus(node) for node in nodes}
    for node in nodes:
        cpus = free_cpus[node]
        gpus = free_gpus[node]
        print(node, "free cpus:", cpus, "free gpus:", gpus)
    reservation = ""
    if free_cpus["jetski"] >= 10 and free_gpus["jetski"] > 0:
        reservation = "--reservation triesch-shared"
    if reservations["triesch-shared"] * 200 < reservations["(null)"]:
        reservation = "--reservation triesch-shared"
    else:
        reservation = ""
    print("reservation: ", reservation)
    return reservation


def additional_args():
    return " hydra.run.dir=" + os.getcwd() + "\\\n"


def get_job_name():
    return os.path.basename(os.getcwd())


def ssh_command(cmd):
    global PASSWORD
    host="fias.uni-frankfurt.de"
    user="wilmot"
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.load_system_host_keys()
    if PASSWORD is None:
        PASSWORD = getpass("Please enter password for the rsa key .ssh/id_rsa\n")
    pkey = RSAKey.from_private_key_file("/home/cwilmot/.ssh/id_rsa", password=PASSWORD)
    client.connect(host, username=user, pkey=pkey)
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

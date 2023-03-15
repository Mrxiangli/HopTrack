import os
import sys
import subprocess
import time

cur_dir = os.getcwd()

if len(sys.argv) != 2:
    assert("Error number of arguments")
mode = sys.argv[1]

if True:
        eng_logfile = '{}/{}_NoTraj_energy.txt'.format(cur_dir, mode)
        p_stats = subprocess.Popen(
            ["sudo tegrastats --logfile {} &".format(eng_logfile)],
            shell=True
        )

        start_exec = subprocess.Popen(
            ["sh autotest_power.sh {} dis_traj".format(mode)],
            shell=True
        )
        start_exec.communicate()

        tegra = subprocess.check_output("pgrep tegrastats",shell=True).decode()
        subprocess.check_output("sudo kill {}".format(tegra),shell=True)

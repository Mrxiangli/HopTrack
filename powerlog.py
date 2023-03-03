import os
import sys
import subprocess
import time

cur_dir = os.getcwd()

if len(sys.argv) != 2:
    assert("Error number of arguments")
mode = sys.argv[1]
if len(sys.argv) == 3:
    traj = "Notraj"
else:
    traj = "traj"

if True:
        eng_logfile = '{}/{}_{}_energy.txt'.format(cur_dir, mode, traj)
        p_stats = subprocess.Popen(
            ["sudo tegrastats --logfile {} &".format(eng_logfile)],
            shell=True
        )

        start_exec = subprocess.Popen(
            ["sh autotest_power.sh {}".format(mode)],
            shell=True
        )
        start_exec.communicate()

        tegra = subprocess.check_output("pgrep tegrastats",shell=True).decode()
        subprocess.check_output("sudo kill {}".format(tegra),shell=True)
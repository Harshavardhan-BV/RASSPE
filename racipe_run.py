#!/usr/bin/env python
import os
import subprocess
import glob
import shutil
import time

## Inputs
# Topofile directory
tpdir = "./TOPO/"
# Number of cores to use, automatically set to the number of cores - 2
numC = os.cpu_count()-2
# Number of parameter sets
num_paras = 10000
# Number of initial conditions
num_ode = 1000

# List the topofiles
tpfl = glob.glob("*.topo", root_dir=tpdir)
tpfl.sort()

# Log to keep track of the commands
rlog = open("racipe_log.txt", "a")

rcpli = []
# Create directories for the results and set up the commands
for t in tpfl:
    for rep in range(1,4):
        # Filename without the extension
        t = os.path.splitext(t)[0]
        if len(t)>70:
            raise ValueError("Topofile name too long, please rename the topofile to less than 10 characters")
        # Make directories for the particular replicate
        os.makedirs("Results/" + t + "/" + str(rep), exist_ok=True)
        # Copy the topofile to the directory
        shutil.copy(tpdir + t + ".topo", "Results/" + t + "/" + str(rep) + "/")
        # Command to run RACIPE
        rcpli.append(f"RACIPE Results/{t}/{rep}/*.topo -num_paras {num_paras} -num_ode {num_ode} -threads {numC} & wait")

for rnRCP in rcpli:
    print(rnRCP)
    # Write the start time & command to the log
    rlog.write(f"{time.ctime()}\t{rnRCP}\t ")
    # Run the command
    subprocess.run(rnRCP, shell=True)
    # Write the end time to the log
    rlog.write(f"{time.ctime()} \n")

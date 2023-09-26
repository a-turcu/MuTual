#!/usr/bin/bash

# https://servicedesk.surf.nl/wiki/pages/viewpage.action?pageId=30660252#JupyterNotebooksonSnellius/Lisa-RunningJupyterNotebooksonLisa/Snelliususingyourownbatchscript

#SBATCH --time 00:30:00
#SBATCH --ntasks 1
#SBATCH --gpus 1
#SBATCH --partition gpu
#SBATCH --job-name=start_jupyter
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

# Make sure the jupyter command is available, either by loading the appropriate modules, sourcing your own virtual environment, etc.
eval "$(micromamba shell hook --shell bash)"
micromamba activate mutual-dl4nlp

# Choose random port and print instructions to connect
PORT=`shuf -i 5000-5999 -n 1`
LOGIN_HOST=${SLURM_SUBMIT_HOST}-pub.snellius.surf.nl
BATCH_HOST=$(hostname)

echo "To connect to the notebook type the following command from your local terminal:"
echo "TERM=xterm-256color ssh -J ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT}:localhost:${PORT}"
echo
echo "After connection is established in your local browser go to the address:"
echo "http://localhost:${PORT}"

jupyter lab --no-browser --port $PORT

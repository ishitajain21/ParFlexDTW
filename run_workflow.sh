#!/bin/bash

# 1. Sync with Git
echo "Syncing with Git..."
git add .
git commit -m "Automated update: $(date)"

# If push fails, stop the script
git push || { echo "Git push failed! Notebook will not run."; exit 1; }

# 2. Run the Notebook (Using the direct path to bypass Conda manager errors)
echo "Starting Jupyter execution..."

# We use the full path to the jupyter executable found in your traceback
nohup /home/ijain/ttmp/anaconda3/envs/mir/bin/jupyter nbconvert \
    --to notebook \
    --execute 04_Align_Benchmarks_sparse_starts.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    > run.log 2>&1 &

echo "Process started. Check run.log for progress."#!/bin/bash

# 1. Git operations
# This adds all changes, commits with a timestamp, and pushes
echo "Syncing with Git..."
git add .
git commit -m "Automated update: $(date)"
git push

# 2. Run the Jupyter Notebook in the background
# We use 'conda run -n mir' to ensure it uses the correct environment
echo "Starting Jupyter execution in the 'mir' environment..."

nohup conda run -n mir jupyter nbconvert \
    --to notebook \
    --execute 04_Align_Benchmarks_sparse_starts.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    > run.log 2>&1 &

echo "Process started in the background. Check run.log for updates."

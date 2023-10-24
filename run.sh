#!/bin/bash

# Get the command and its arguments as input
task=$1

echo "Running task: $task"

# Execute the command with its arguments
echo "Executing Input Reduction..."
python -m explainer.input_reduction_ig_approx --task $task
echo "Calculating Generality..."
python -m explainer.calculate_generality --task $task
echo "Identifying Shortcut Reasoning..."
python -m explainer.check --task $task

echo "Done! Check the results in the results at ./explainer/examples/task/figs/{task}.csv"

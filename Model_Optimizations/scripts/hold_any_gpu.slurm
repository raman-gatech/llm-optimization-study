#!/bin/bash
#SBATCH --job-name=cs6675
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -C 'nvidia-gpu'
#SBATCH --time=16:00:00
#SBATCH --dependency=singleton

tail -f /dev/null

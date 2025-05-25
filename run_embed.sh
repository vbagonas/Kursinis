#!/bin/bash
#SBATCH -p main
#SBATCH -n8
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00:00
#SBATCH -J embed_dump
#SBATCH -o embed_dump_%j.out
#SBATCH -e embed_dump_%j.err

cd ~/kursinis_milv/
source kursinis-env/bin/activate

python3 embed_func.py